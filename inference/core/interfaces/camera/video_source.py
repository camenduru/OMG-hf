import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, List, Optional, Protocol, Union

import cv2
import supervision as sv

from inference.core import logger
from inference.core.env import (
    DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE,
    DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW,
    DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES,
)
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
)
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    SourceConnectionError,
    StreamOperationNotAllowedError,
)

VIDEO_SOURCE_CONTEXT = "video_source"
VIDEO_CONSUMER_CONTEXT = "video_consumer"
SOURCE_STATE_UPDATE_EVENT = "SOURCE_STATE_UPDATE"
SOURCE_ERROR_EVENT = "SOURCE_ERROR"
FRAME_CAPTURED_EVENT = "FRAME_CAPTURED"
FRAME_DROPPED_EVENT = "FRAME_DROPPED"
FRAME_CONSUMED_EVENT = "FRAME_CONSUMED"
VIDEO_CONSUMPTION_STARTED_EVENT = "VIDEO_CONSUMPTION_STARTED"
VIDEO_CONSUMPTION_FINISHED_EVENT = "VIDEO_CONSUMPTION_FINISHED"


class StreamState(Enum):
    NOT_STARTED = "NOT_STARTED"
    INITIALISING = "INITIALISING"
    RESTARTING = "RESTARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    MUTED = "MUTED"
    TERMINATING = "TERMINATING"
    ENDED = "ENDED"
    ERROR = "ERROR"


START_ELIGIBLE_STATES = {
    StreamState.NOT_STARTED,
    StreamState.RESTARTING,
    StreamState.ENDED,
}
PAUSE_ELIGIBLE_STATES = {StreamState.RUNNING}
MUTE_ELIGIBLE_STATES = {StreamState.RUNNING}
RESUME_ELIGIBLE_STATES = {StreamState.PAUSED, StreamState.MUTED}
TERMINATE_ELIGIBLE_STATES = {
    StreamState.MUTED,
    StreamState.RUNNING,
    StreamState.PAUSED,
    StreamState.RESTARTING,
    StreamState.ENDED,
    StreamState.ERROR,
}
RESTART_ELIGIBLE_STATES = {
    StreamState.MUTED,
    StreamState.RUNNING,
    StreamState.PAUSED,
    StreamState.ENDED,
    StreamState.ERROR,
}


class BufferFillingStrategy(Enum):
    WAIT = "WAIT"
    DROP_OLDEST = "DROP_OLDEST"
    ADAPTIVE_DROP_OLDEST = "ADAPTIVE_DROP_OLDEST"
    DROP_LATEST = "DROP_LATEST"
    ADAPTIVE_DROP_LATEST = "ADAPTIVE_DROP_LATEST"


ADAPTIVE_STRATEGIES = {
    BufferFillingStrategy.ADAPTIVE_DROP_LATEST,
    BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
}
DROP_OLDEST_STRATEGIES = {
    BufferFillingStrategy.DROP_OLDEST,
    BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
}


class BufferConsumptionStrategy(Enum):
    LAZY = "LAZY"
    EAGER = "EAGER"


@dataclass(frozen=True)
class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float


@dataclass(frozen=True)
class SourceMetadata:
    source_properties: Optional[SourceProperties]
    source_reference: str
    buffer_size: int
    state: StreamState
    buffer_filling_strategy: Optional[BufferFillingStrategy]
    buffer_consumption_strategy: Optional[BufferConsumptionStrategy]


class VideoSourceMethod(Protocol):
    def __call__(self, video_source: "VideoSource", *args, **kwargs) -> None: ...


def lock_state_transition(
    method: VideoSourceMethod,
) -> Callable[["VideoSource"], None]:
    def locked_executor(video_source: "VideoSource", *args, **kwargs) -> None:
        with video_source._state_change_lock:
            return method(video_source, *args, **kwargs)

    return locked_executor


class VideoSource:
    @classmethod
    def init(
        cls,
        video_reference: Union[str, int],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        adaptive_mode_stream_pace_tolerance: float = DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE,
        adaptive_mode_reader_pace_tolerance: float = DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE,
        minimum_adaptive_mode_samples: int = DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES,
        maximum_adaptive_frames_dropped_in_row: int = DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW,
    ):
        """
        This class is meant to represent abstraction over video sources - both video files and
        on-line streams that are possible to be consumed and used by other components of `inference`
        library.

        Before digging into details of the class behaviour, it is advised to familiarise with the following
        concepts and implementation assumptions:

        1. Video file can be accessed from local (or remote) storage by the consumer in a pace dictated by
            its processing capabilities. If processing is faster than the frame rate of video, operations
            may be executed in a time shorter than the time of video playback. In the opposite case - consumer
            may freely decode and process frames in its own pace, without risk for failures due to temporal
            dependencies of processing - this is classical offline processing example.
        2. Video streams, on the other hand, usually need to be consumed in a pace near to their frame-rate -
            in other words - this is on-line processing example. Consumer being faster than incoming stream
            frames cannot utilise its resources to the full extent as not-yet-delivered data would be needed.
            Slow consumer, however, may not be able to process everything on time and to keep up with the pace
            of stream - some frames would need to be dropped. Otherwise - over time, consumer could go out of
            sync with the stream causing decoding failures or unpredictable behavior.

        To fit those two types of video sources, `VideoSource` introduces the concept of buffered decoding of
        video stream (like at the YouTube - player buffers some frames that are soon to be displayed).
        The way on how buffer is filled and consumed dictates the behavior of `VideoSource`.

        Starting from `BufferFillingStrategy` - we have 3 basic options:
        * WAIT: in case of slow video consumption, when buffer is full - `VideoSource` will wait for
        the empty spot in buffer before next frame will be processed - this is suitable in cases when
        we want to ensure EACH FRAME of the video to be processed
        * DROP_OLDEST: when buffer is full, the frame that sits there for the longest time will be dropped -
        this is suitable for cases when we want to process the most recent frames possible
        * DROP_LATEST: when buffer is full, the newly decoded frame is dropped - useful in cases when
        it is expected to have processing performance drops, but we would like to consume portions of
        video that are locally smooth - but this is probably the least common use-case.

        On top of that - there are two ADAPTIVE strategies: ADAPTIVE_DROP_OLDEST and ADAPTIVE_DROP_LATEST,
        which are equivalent to DROP_OLDEST and DROP_LATEST with adaptive decoding feature enabled. The notion
        of that mode will be described later.

        Naturally, decoded frames must also be consumed. `VideoSource` provides a handy interface for reading
        a video source frames by a SINGLE consumer. Consumption strategy can also be dictated via
        `BufferConsumptionStrategy`:
        * LAZY - consume all the frames from decoding buffer one-by-one
        * EAGER - at each readout - take all frames already buffered, drop all of them apart from the most recent

        In consequence - there are various combinations of `BufferFillingStrategy` and `BufferConsumptionStrategy`.
        The most popular would be:
        * `BufferFillingStrategy.WAIT` and `BufferConsumptionStrategy.LAZY` - to always decode and process each and
            every frame of the source (useful while processing video files - and default behaviour enforced by
            `inference` if there is no explicit configuration)
        * `BufferFillingStrategy.DROP_OLDEST` and `BufferConsumptionStrategy.EAGER` - to always process the most
            recent frames of source (useful while processing video streams when low latency [real-time experience]
            is required - ADAPTIVE version of this is default for streams)

        ADAPTIVE strategies were introduced to handle corner-cases, when consumer hardware is not capable to consume
        video stream and process frames at the same time (for instance - Nvidia Jetson devices running processing
        against hi-res streams with high FPS ratio). It acts with buffer in nearly the same way as `DROP_OLDEST`
        and `DROP_LATEST` strategies, but there are two more conditions that may influence frame drop:
        * announced rate of source - which in fact dictate the pace of frames grabbing from incoming stream that
        MUST be met by consumer to avoid strange decoding issues causing decoder to fail - if the pace of frame grabbing
        deviates too much - decoding will be postponed, and frames dropped to grab next ones sooner
        * consumption rate - in resource constraints environment, not only decoding is problematic from the performance
        perspective - but also heavy processing. If consumer is not quick enough - allocating more useful resources
        for decoding frames that may never be processed is a waste. That's why - if decoding happens more frequently
        than consumption of frame - ADAPTIVE mode causes decoding to be done in a slower pace and more frames are just
        grabbed and dropped on the floor.
        ADAPTIVE mode increases latency slightly, but may be the only way to operate in some cases.
        Behaviour of adaptive mode, including the maximum acceptable deviations of frames grabbing pace from source,
        reader pace and maximum number of consecutive frames dropped in ADAPTIVE mode are configurable by clients,
        with reasonable defaults being set.

        `VideoSource` emits events regarding its activity - which can be intercepted by custom handlers. Take
        into account that they are always executed in context of thread invoking them (and should be fast to complete,
        otherwise may block the flow of stream consumption). All errors raised will be emitted as logger warnings only.

        `VideoSource` implementation is naturally multithreading, with different thread decoding video and different
        one consuming it and manipulating source state. Implementation of user interface is thread-safe, although
        stream it is meant to be consumed by a single thread only.

        ENV variables involved:
        * VIDEO_SOURCE_BUFFER_SIZE - default: 64
        * VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE - default: 0.1
        * VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE - default: 5.0
        * VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES - default: 10
        * VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW - default: 16

        As an `inference` user, please use .init() method instead of constructor to instantiate objects.

        Args:
            video_reference (Union[str, int]): Either str with file or stream reference, or int representing device ID
            buffer_size (int): size of decoding buffer
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers for status updates
            buffer_filling_strategy (Optional[BufferFillingStrategy]): Settings for buffer filling strategy - if not
                given - automatic choice regarding source type will be applied
            buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Settings for buffer consumption strategy,
                if not given - automatic choice regarding source type will be applied
            adaptive_mode_stream_pace_tolerance (float): Maximum deviation between frames grabbing pace and stream pace
                that will not trigger adaptive mode frame drop
            adaptive_mode_reader_pace_tolerance (float): Maximum deviation between decoding pace and stream consumption
                pace that will not trigger adaptive mode frame drop
            minimum_adaptive_mode_samples (int): Minimal number of frames to be used to establish actual pace of
                processing, before adaptive mode can drop any frame
            maximum_adaptive_frames_dropped_in_row (int): Maximum number of frames dropped in row due to application of
                adaptive strategy

        Returns: Instance of `VideoSource` class
        """
        frames_buffer = Queue(maxsize=buffer_size)
        if status_update_handlers is None:
            status_update_handlers = []
        video_consumer = VideoConsumer.init(
            buffer_filling_strategy=buffer_filling_strategy,
            adaptive_mode_stream_pace_tolerance=adaptive_mode_stream_pace_tolerance,
            adaptive_mode_reader_pace_tolerance=adaptive_mode_reader_pace_tolerance,
            minimum_adaptive_mode_samples=minimum_adaptive_mode_samples,
            maximum_adaptive_frames_dropped_in_row=maximum_adaptive_frames_dropped_in_row,
            status_update_handlers=status_update_handlers,
        )
        return cls(
            stream_reference=video_reference,
            frames_buffer=frames_buffer,
            status_update_handlers=status_update_handlers,
            buffer_consumption_strategy=buffer_consumption_strategy,
            video_consumer=video_consumer,
        )

    def __init__(
        self,
        stream_reference: Union[str, int],
        frames_buffer: Queue,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
        video_consumer: "VideoConsumer",
    ):
        self._stream_reference = stream_reference
        self._video: Optional[cv2.VideoCapture] = None
        self._source_properties: Optional[SourceProperties] = None
        self._frames_buffer = frames_buffer
        self._status_update_handlers = status_update_handlers
        self._buffer_consumption_strategy = buffer_consumption_strategy
        self._video_consumer = video_consumer
        self._state = StreamState.NOT_STARTED
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._stream_consumption_thread: Optional[Thread] = None
        self._state_change_lock = Lock()

    @lock_state_transition
    def restart(self, wait_on_frames_consumption: bool = True) -> None:
        """
        Method to restart source consumption. Eligible to be used in states:
        [MUTED, RUNNING, PAUSED, ENDED, ERROR].
        End state:
        * INITIALISING - that should change into RUNNING once first frame is ready to be grabbed
        * ERROR - if it was not possible to connect with source

        Thread safe - only one transition of states possible at the time.

        Args:
            wait_on_frames_consumption (bool): Flag telling if all frames from buffer must be consumed before
                completion of this operation.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
            * SourceConnectionError: if source cannot be connected
        """
        if self._state not in RESTART_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESTART stream in state: {self._state}"
            )
        self._restart(wait_on_frames_consumption=wait_on_frames_consumption)

    @lock_state_transition
    def start(self) -> None:
        """
        Method to be used to start source consumption. Eligible to be used in states:
        [NOT_STARTED, ENDED, (RESTARTING - which is internal state only)]
        End state:
        * INITIALISING - that should change into RUNNING once first frame is ready to be grabbed
        * ERROR - if it was not possible to connect with source

        Thread safe - only one transition of states possible at the time.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
            * SourceConnectionError: if source cannot be connected
        """
        if self._state not in START_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not START stream in state: {self._state}"
            )
        self._start()

    @lock_state_transition
    def terminate(self, wait_on_frames_consumption: bool = True) -> None:
        """
        Method to be used to terminate source consumption. Eligible to be used in states:
        [MUTED, RUNNING, PAUSED, ENDED, ERROR, (RESTARTING - which is internal state only)]
        End state:
        * ENDED - indicating success of the process
        * ERROR - if error with processing occurred

        Must be used to properly dispose resources at the end.

        Thread safe - only one transition of states possible at the time.

        Args:
            wait_on_frames_consumption (bool): Flag telling if all frames from buffer must be consumed before
                completion of this operation.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
        """
        if self._state not in TERMINATE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not TERMINATE stream in state: {self._state}"
            )
        self._terminate(wait_on_frames_consumption=wait_on_frames_consumption)

    @lock_state_transition
    def pause(self) -> None:
        """
        Method to be used to pause source consumption. During pause - no new frames are consumed.
        Used on on-line streams for too long may cause stream disconnection.
        Eligible to be used in states:
        [RUNNING]
        End state:
        * PAUSED

        Thread safe - only one transition of states possible at the time.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
        """
        if self._state not in PAUSE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not PAUSE stream in state: {self._state}"
            )
        self._pause()

    @lock_state_transition
    def mute(self) -> None:
        """
        Method to be used to mute source consumption. Muting is an equivalent of pause for stream - where
        frames grabbing is not put on hold, just new frames decoding and buffering is not allowed - causing
        intermediate frames to be dropped. May be also used against files, although arguably less useful.
        Eligible to be used in states:
        [RUNNING]
        End state:
        * MUTED

        Thread safe - only one transition of states possible at the time.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
        """
        if self._state not in MUTE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not MUTE stream in state: {self._state}"
            )
        self._mute()

    @lock_state_transition
    def resume(self) -> None:
        """
        Method to recover from pause or mute into running state.
        [PAUSED, MUTED]
        End state:
        * RUNNING

        Thread safe - only one transition of states possible at the time.

        Returns: None
        Throws:
            * StreamOperationNotAllowedError: if executed in context of incorrect state of the source
        """
        if self._state not in RESUME_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESUME stream in state: {self._state}"
            )
        self._resume()

    def get_state(self) -> StreamState:
        """
        Method to get current state of the `VideoSource`

        Returns: StreamState
        """
        return self._state

    def frame_ready(self) -> bool:
        """
        Method to check if decoded frame is ready for consumer

        Returns: boolean flag indicating frame readiness
        """
        return not self._frames_buffer.empty()

    def read_frame(self) -> VideoFrame:
        """
        Method to be used by the consumer to get decoded source frame.

        Returns: VideoFrame object with decoded frame and its metadata.
        Throws:
            * EndOfStreamError: when trying to get the frame from closed source.
        """
        if self._buffer_consumption_strategy is BufferConsumptionStrategy.EAGER:
            video_frame: Optional[VideoFrame] = purge_queue(
                queue=self._frames_buffer,
                on_successful_read=self._video_consumer.notify_frame_consumed,
            )
        else:
            video_frame: Optional[VideoFrame] = self._frames_buffer.get()
            self._frames_buffer.task_done()
            self._video_consumer.notify_frame_consumed()
        if video_frame is None:
            raise EndOfStreamError(
                "Attempted to retrieve frame from stream that already ended."
            )
        send_video_source_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CONSUMED_EVENT,
            payload={
                "frame_timestamp": video_frame.frame_timestamp,
                "frame_id": video_frame.frame_id,
            },
            status_update_handlers=self._status_update_handlers,
        )
        return video_frame

    def describe_source(self) -> SourceMetadata:
        return SourceMetadata(
            source_properties=self._source_properties,
            source_reference=self._stream_reference,
            buffer_size=self._frames_buffer.maxsize,
            state=self._state,
            buffer_filling_strategy=self._video_consumer.buffer_filling_strategy,
            buffer_consumption_strategy=self._buffer_consumption_strategy,
        )

    def _restart(self, wait_on_frames_consumption: bool = True) -> None:
        self._terminate(wait_on_frames_consumption=wait_on_frames_consumption)
        self._change_state(target_state=StreamState.RESTARTING)
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._video: Optional[cv2.VideoCapture] = None
        self._source_properties: Optional[SourceProperties] = None
        self._start()

    def _start(self) -> None:
        self._change_state(target_state=StreamState.INITIALISING)
        self._video = cv2.VideoCapture(self._stream_reference)
        if not self._video.isOpened():
            self._change_state(target_state=StreamState.ERROR)
            raise SourceConnectionError(
                f"Cannot connect to video source under reference: {self._stream_reference}"
            )
        self._source_properties = discover_source_properties(stream=self._video)
        self._video_consumer.reset(source_properties=self._source_properties)
        if self._source_properties.is_file:
            self._set_file_mode_consumption_strategies()
        else:
            self._set_stream_mode_consumption_strategies()
        self._playback_allowed.set()
        self._stream_consumption_thread = Thread(target=self._consume_video)
        self._stream_consumption_thread.start()

    def _terminate(self, wait_on_frames_consumption: bool) -> None:
        if self._state in RESUME_ELIGIBLE_STATES:
            self._resume()
        previous_state = self._state
        self._change_state(target_state=StreamState.TERMINATING)
        if self._stream_consumption_thread is not None:
            self._stream_consumption_thread.join()
        if wait_on_frames_consumption:
            self._frames_buffer.join()
        if previous_state is not StreamState.ERROR:
            self._change_state(target_state=StreamState.ENDED)

    def _pause(self) -> None:
        self._playback_allowed.clear()
        self._change_state(target_state=StreamState.PAUSED)

    def _mute(self) -> None:
        self._frames_buffering_allowed = False
        self._change_state(target_state=StreamState.MUTED)

    def _resume(self) -> None:
        previous_state = self._state
        self._change_state(target_state=StreamState.RUNNING)
        if previous_state is StreamState.PAUSED:
            self._video_consumer.reset_stream_consumption_pace()
            self._playback_allowed.set()
        if previous_state is StreamState.MUTED:
            self._frames_buffering_allowed = True

    def _set_file_mode_consumption_strategies(self) -> None:
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.LAZY

    def _set_stream_mode_consumption_strategies(self) -> None:
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.EAGER

    def _consume_video(self) -> None:
        send_video_source_status_update(
            severity=UpdateSeverity.INFO,
            event_type=VIDEO_CONSUMPTION_STARTED_EVENT,
            status_update_handlers=self._status_update_handlers,
        )
        logger.info(f"Video consumption started")
        try:
            self._change_state(target_state=StreamState.RUNNING)
            declared_source_fps = None
            if self._source_properties is not None:
                declared_source_fps = self._source_properties.fps
            while self._video.isOpened():
                if self._state is StreamState.TERMINATING:
                    break
                self._playback_allowed.wait()
                success = self._video_consumer.consume_frame(
                    video=self._video,
                    declared_source_fps=declared_source_fps,
                    buffer=self._frames_buffer,
                    frames_buffering_allowed=self._frames_buffering_allowed,
                )
                if not success:
                    break
            self._frames_buffer.put(None)
            self._video.release()
            self._change_state(target_state=StreamState.ENDED)
            send_video_source_status_update(
                severity=UpdateSeverity.INFO,
                event_type=VIDEO_CONSUMPTION_FINISHED_EVENT,
                status_update_handlers=self._status_update_handlers,
            )
            logger.info(f"Video consumption finished")
        except Exception as error:
            self._change_state(target_state=StreamState.ERROR)
            payload = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "error_context": "stream_consumer_thread",
            }
            send_video_source_status_update(
                severity=UpdateSeverity.ERROR,
                event_type=SOURCE_ERROR_EVENT,
                payload=payload,
                status_update_handlers=self._status_update_handlers,
            )
            logger.exception("Encountered error in video consumption thread")

    def _change_state(self, target_state: StreamState) -> None:
        payload = {
            "previous_state": self._state,
            "new_state": target_state,
        }
        self._state = target_state
        send_video_source_status_update(
            severity=UpdateSeverity.INFO,
            event_type=SOURCE_STATE_UPDATE_EVENT,
            payload=payload,
            status_update_handlers=self._status_update_handlers,
        )

    def __iter__(self) -> "VideoSource":
        return self

    def __next__(self) -> VideoFrame:
        """
        Method allowing to use `VideoSource` convenient to read frames

        Returns: VideoFrame

        Example:
            ```python
            source = VideoSource.init(video_reference="./some.mp4")
            source.start()

            for frame in source:
                 pass
            ```
        """
        try:
            return self.read_frame()
        except EndOfStreamError:
            raise StopIteration()


class VideoConsumer:
    """
    This class should be consumed as part of internal implementation.
    It provides abstraction around stream consumption strategies.
    """

    @classmethod
    def init(
        cls,
        buffer_filling_strategy: Optional[BufferFillingStrategy],
        adaptive_mode_stream_pace_tolerance: float,
        adaptive_mode_reader_pace_tolerance: float,
        minimum_adaptive_mode_samples: int,
        maximum_adaptive_frames_dropped_in_row: int,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
    ) -> "VideoConsumer":
        minimum_adaptive_mode_samples = max(minimum_adaptive_mode_samples, 2)
        reader_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        stream_consumption_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        decoding_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        return cls(
            buffer_filling_strategy=buffer_filling_strategy,
            adaptive_mode_stream_pace_tolerance=adaptive_mode_stream_pace_tolerance,
            adaptive_mode_reader_pace_tolerance=adaptive_mode_reader_pace_tolerance,
            minimum_adaptive_mode_samples=minimum_adaptive_mode_samples,
            maximum_adaptive_frames_dropped_in_row=maximum_adaptive_frames_dropped_in_row,
            status_update_handlers=status_update_handlers,
            reader_pace_monitor=reader_pace_monitor,
            stream_consumption_pace_monitor=stream_consumption_pace_monitor,
            decoding_pace_monitor=decoding_pace_monitor,
        )

    def __init__(
        self,
        buffer_filling_strategy: Optional[BufferFillingStrategy],
        adaptive_mode_stream_pace_tolerance: float,
        adaptive_mode_reader_pace_tolerance: float,
        minimum_adaptive_mode_samples: int,
        maximum_adaptive_frames_dropped_in_row: int,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        reader_pace_monitor: sv.FPSMonitor,
        stream_consumption_pace_monitor: sv.FPSMonitor,
        decoding_pace_monitor: sv.FPSMonitor,
    ):
        self._buffer_filling_strategy = buffer_filling_strategy
        self._frame_counter = 0
        self._adaptive_mode_stream_pace_tolerance = adaptive_mode_stream_pace_tolerance
        self._adaptive_mode_reader_pace_tolerance = adaptive_mode_reader_pace_tolerance
        self._minimum_adaptive_mode_samples = minimum_adaptive_mode_samples
        self._maximum_adaptive_frames_dropped_in_row = (
            maximum_adaptive_frames_dropped_in_row
        )
        self._adaptive_frames_dropped_in_row = 0
        self._reader_pace_monitor = reader_pace_monitor
        self._stream_consumption_pace_monitor = stream_consumption_pace_monitor
        self._decoding_pace_monitor = decoding_pace_monitor
        self._status_update_handlers = status_update_handlers

    @property
    def buffer_filling_strategy(self) -> Optional[BufferFillingStrategy]:
        return self._buffer_filling_strategy

    def reset(self, source_properties: SourceProperties) -> None:
        if source_properties.is_file:
            self._set_file_mode_buffering_strategies()
        else:
            self._set_stream_mode_buffering_strategies()
        self._reader_pace_monitor.reset()
        self.reset_stream_consumption_pace()
        self._decoding_pace_monitor.reset()
        self._adaptive_frames_dropped_in_row = 0

    def reset_stream_consumption_pace(self) -> None:
        self._stream_consumption_pace_monitor.reset()

    def notify_frame_consumed(self) -> None:
        self._reader_pace_monitor.tick()

    def consume_frame(
        self,
        video: cv2.VideoCapture,
        declared_source_fps: Optional[float],
        buffer: Queue,
        frames_buffering_allowed: bool,
    ) -> bool:
        frame_timestamp = datetime.now()
        success = video.grab()
        self._stream_consumption_pace_monitor.tick()
        if not success:
            return False
        self._frame_counter += 1
        send_video_source_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CAPTURED_EVENT,
            payload={
                "frame_timestamp": frame_timestamp,
                "frame_id": self._frame_counter,
            },
            status_update_handlers=self._status_update_handlers,
        )
        return self._consume_stream_frame(
            video=video,
            declared_source_fps=declared_source_fps,
            frame_timestamp=frame_timestamp,
            buffer=buffer,
            frames_buffering_allowed=frames_buffering_allowed,
        )

    def _set_file_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.WAIT

    def _set_stream_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.ADAPTIVE_DROP_OLDEST

    def _consume_stream_frame(
        self,
        video: cv2.VideoCapture,
        declared_source_fps: Optional[float],
        frame_timestamp: datetime,
        buffer: Queue,
        frames_buffering_allowed: bool,
    ) -> bool:
        """
        Returns: boolean flag with success status
        """
        if not frames_buffering_allowed:
            send_frame_drop_update(
                frame_timestamp=frame_timestamp,
                frame_id=self._frame_counter,
                cause="Buffering not allowed at the moment",
                status_update_handlers=self._status_update_handlers,
            )
            return True
        if self._frame_should_be_adaptively_dropped(
            declared_source_fps=declared_source_fps
        ):
            self._adaptive_frames_dropped_in_row += 1
            send_frame_drop_update(
                frame_timestamp=frame_timestamp,
                frame_id=self._frame_counter,
                cause="ADAPTIVE strategy",
                status_update_handlers=self._status_update_handlers,
            )
            return True
        self._adaptive_frames_dropped_in_row = 0
        if (
            not buffer.full()
            or self._buffer_filling_strategy is BufferFillingStrategy.WAIT
        ):
            return decode_video_frame_to_buffer(
                frame_timestamp=frame_timestamp,
                frame_id=self._frame_counter,
                video=video,
                buffer=buffer,
                decoding_pace_monitor=self._decoding_pace_monitor,
            )
        if self._buffer_filling_strategy in DROP_OLDEST_STRATEGIES:
            return self._process_stream_frame_dropping_oldest(
                frame_timestamp=frame_timestamp,
                video=video,
                buffer=buffer,
            )
        send_frame_drop_update(
            frame_timestamp=frame_timestamp,
            frame_id=self._frame_counter,
            cause="DROP_LATEST strategy",
            status_update_handlers=self._status_update_handlers,
        )
        return True

    def _frame_should_be_adaptively_dropped(
        self, declared_source_fps: Optional[float]
    ) -> bool:
        if self._buffer_filling_strategy not in ADAPTIVE_STRATEGIES:
            return False
        if (
            self._adaptive_frames_dropped_in_row
            >= self._maximum_adaptive_frames_dropped_in_row
        ):
            return False
        if (
            len(self._stream_consumption_pace_monitor.all_timestamps)
            <= self._minimum_adaptive_mode_samples
        ):
            # not enough observations
            return False
        stream_consumption_pace = self._stream_consumption_pace_monitor()
        announced_stream_fps = stream_consumption_pace
        if declared_source_fps is not None and declared_source_fps > 0:
            announced_stream_fps = declared_source_fps
        if (
            announced_stream_fps - stream_consumption_pace
            > self._adaptive_mode_stream_pace_tolerance
        ):
            # cannot keep up with stream emission
            return True
        if (
            len(self._reader_pace_monitor.all_timestamps)
            <= self._minimum_adaptive_mode_samples
        ) or (
            len(self._decoding_pace_monitor.all_timestamps)
            <= self._minimum_adaptive_mode_samples
        ):
            # not enough observations
            return False
        actual_reader_pace = get_fps_if_tick_happens_now(
            fps_monitor=self._reader_pace_monitor
        )
        decoding_pace = self._decoding_pace_monitor()
        if (
            decoding_pace - actual_reader_pace
            > self._adaptive_mode_reader_pace_tolerance
        ):
            # we are too fast for the reader - time to save compute on decoding
            return True
        return False

    def _process_stream_frame_dropping_oldest(
        self,
        frame_timestamp: datetime,
        video: cv2.VideoCapture,
        buffer: Queue,
    ) -> bool:
        drop_single_frame_from_buffer(
            buffer=buffer,
            cause="DROP_OLDEST strategy",
            status_update_handlers=self._status_update_handlers,
        )
        return decode_video_frame_to_buffer(
            frame_timestamp=frame_timestamp,
            frame_id=self._frame_counter,
            video=video,
            buffer=buffer,
            decoding_pace_monitor=self._decoding_pace_monitor,
        )


def discover_source_properties(stream: cv2.VideoCapture) -> SourceProperties:
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return SourceProperties(
        width=width,
        height=height,
        total_frames=total_frames,
        is_file=total_frames > 0,
        fps=fps,
    )


def purge_queue(
    queue: Queue,
    wait_on_empty: bool = True,
    on_successful_read: Callable[[], None] = lambda: None,
) -> Optional[Any]:
    result = None
    if queue.empty() and wait_on_empty:
        result = queue.get()
        queue.task_done()
        on_successful_read()
    while not queue.empty():
        result = queue.get()
        queue.task_done()
        on_successful_read()
    return result


def drop_single_frame_from_buffer(
    buffer: Queue,
    cause: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
) -> None:
    try:
        video_frame = buffer.get_nowait()
        buffer.task_done()
        send_frame_drop_update(
            frame_timestamp=video_frame.frame_timestamp,
            frame_id=video_frame.frame_id,
            cause=cause,
            status_update_handlers=status_update_handlers,
        )
    except Empty:
        # buffer may be emptied in the meantime, hence we ignore Empty
        pass


def send_frame_drop_update(
    frame_timestamp: datetime,
    frame_id: int,
    cause: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
) -> None:
    send_video_source_status_update(
        severity=UpdateSeverity.DEBUG,
        event_type=FRAME_DROPPED_EVENT,
        payload={
            "frame_timestamp": frame_timestamp,
            "frame_id": frame_id,
            "cause": cause,
        },
        status_update_handlers=status_update_handlers,
        sub_context=VIDEO_CONSUMER_CONTEXT,
    )


def send_video_source_status_update(
    severity: UpdateSeverity,
    event_type: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
    sub_context: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    if payload is None:
        payload = {}
    context = VIDEO_SOURCE_CONTEXT
    if sub_context is not None:
        context = f"{context}.{sub_context}"
    status_update = StatusUpdate(
        timestamp=datetime.now(),
        severity=severity,
        event_type=event_type,
        payload=payload,
        context=context,
    )
    for handler in status_update_handlers:
        try:
            handler(status_update)
        except Exception as error:
            logger.warning(f"Could not execute handler update. Cause: {error}")


def decode_video_frame_to_buffer(
    frame_timestamp: datetime,
    frame_id: int,
    video: cv2.VideoCapture,
    buffer: Queue,
    decoding_pace_monitor: sv.FPSMonitor,
) -> bool:
    success, image = video.retrieve()
    if not success:
        return False
    decoding_pace_monitor.tick()
    video_frame = VideoFrame(
        image=image, frame_id=frame_id, frame_timestamp=frame_timestamp
    )
    buffer.put(video_frame)
    return True


def get_fps_if_tick_happens_now(fps_monitor: sv.FPSMonitor) -> float:
    if len(fps_monitor.all_timestamps) == 0:
        return 0.0
    min_reader_timestamp = fps_monitor.all_timestamps[0]
    now = time.monotonic()
    reader_taken_time = now - min_reader_timestamp
    return (len(fps_monitor.all_timestamps) + 1) / reader_taken_time
