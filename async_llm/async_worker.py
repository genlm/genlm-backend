import asyncio
import weakref
import logging
from abc import ABC, abstractmethod

"""
The AsyncWorker implements a pattern for handling asynchronous request processing. 

It uses two key components: 
    1. A RequestTracker for managing incoming requests and their corresponding futures.
    2. A background processing loop that handles the actual work.

The request handling flow works like this: 
    * When a request is added through add_request(), a future is returned that will eventually contain the result. 
    * The request is queued in the RequestTracker, which maintains a queue of new requests. 
    * The background loop continuously monitors for new requests and, when present, processes them in batches through the abstract batch_process_requests() method.

Subclasses must minimally implement the batch_process_requests() method.
"""

logger = logging.getLogger(__name__)

class RequestTracker:
    """Tracks async requests and their corresponding futures."""

    def __init__(self):
        self._pending_requests = {}
        self._new_requests = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def add_request(self, request_id, request):
        """Add a new request and return a future for the response.
        
        Args:
            request_id: Unique identifier for this request
            request: The request data to be processed
            
        Returns:
            asyncio.Future: A future that will contain the request's result
        """
        if request_id in self._pending_requests:
            raise KeyError(f"Request {request_id} already exists")

        future = asyncio.Future()
        self._pending_requests[request_id] = future
        self._new_requests.put_nowait((request_id, request))
        self.new_requests_event.set()
        logger.debug(f"Added new request with ID: {request_id}")
        return future

    async def wait_for_new_requests(self):
        """Wait for new requests to arrive."""
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self) -> bool:
        return not self._new_requests.empty()


class RequestCounter:
    def __init__(self, start=0):
        self.counter = start

    def __next__(self):
        i = self.counter
        self.counter += 1
        return i

    def reset(self):
        self.counter = 0


class AsyncWorker(ABC):
    """Generic async worker that handles request processing in background."""
    def __init__(self):
        self.request_counter = RequestCounter()
        self.request_tracker = None
        self.background_loop = None
        self.errored_with = None

    @property
    def is_running(self):
        return (
            self.background_loop is not None and
            not self.background_loop.done()
        )

    @property
    def errored(self):
        return self.errored_with is not None

    def start_background_loop(self):
        """Start the background processing loop.
        
        Creates a new background task using a weak reference to self, allowing
        the worker to be garbage collected even while the task is running.
        
        Raises:
            RuntimeError: If the background loop is already running or has errored
        """
        if self.errored:
            raise RuntimeError("Background loop has errored already.") from self.errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")

        # Initialize the RequestTracker here so it uses the right event loop.
        self.request_tracker = RequestTracker()
        self.background_loop = asyncio.get_event_loop().create_task(
            self.run_background_loop(weakref.ref(self))
        )

    def shutdown_background_loop(self):
        """Shut down the background loop."""
        if self.background_loop is not None:
            self.background_loop.cancel()
            self.background_loop = None

    async def add_request(self, request_id, request):
        if not self.is_running:
            self.start_background_loop()
            
        future = self.request_tracker.add_request(request_id, request)
        return await future

    @staticmethod
    async def run_background_loop(worker_ref):
        """Main processing loop that handles requests.
        
        This method runs as a background task and processes incoming requests.
        
        Args:
            worker_ref: Weak reference to the Asyncworker instance
        """
        worker = worker_ref()
        logger.debug("Background loop started")

        while True:
            try:
                # Store local reference to tracker and release worker
                # so that it can be garbage collected. 
                request_tracker = worker.request_tracker
                del worker

                # Check if worker was collected.
                if worker_ref() is None:
                    return

                # Wait for new request event.
                await request_tracker.wait_for_new_requests()
                logger.debug("New requests available for processing")

                # Get fresh reference to worker.
                worker = worker_ref()
                if not worker:
                    return

                await worker.step()
            except Exception as e:
                worker = worker_ref()
                if worker:
                    worker.errored_with = e
                    for future in worker.request_tracker._pending_requests.values():
                        future.set_exception(e)
                    worker.request_tracker._pending_requests.clear()
                raise

    async def step(self):
        """Process a batch of pending requests."""
        requests = []
        request_ids = []
        while self.request_tracker.has_new_requests():
            request_id, request = self.request_tracker._new_requests.get_nowait()
            requests.append(request)
            request_ids.append(request_id)

        if not requests:
            return

        logger.debug(f"Processing batch of {len(requests)} requests")
        try:
            results = await self.batch_process_requests(requests)
            for request_id, result in zip(request_ids, results):
                future = self.request_tracker._pending_requests.pop(request_id)
                future.set_result(result)
        except Exception as e:
            for request_id in request_ids:
                future = self.request_tracker._pending_requests.pop(request_id, None)
                if future:
                    future.set_exception(e)
            raise

    @abstractmethod
    async def batch_process_requests(self, requests):
        """Process a batch of requests.
        
        This method must be implemented by subclasses to define how
        to process a batch of requests.
        
        Args:
            requests: List of requests to process
            
        Returns:
            List of results corresponding to the input requests
        """
        pass

    def __del__(self):
        self.shutdown_background_loop()