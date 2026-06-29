import queue
import threading
from collections import defaultdict


_subscribers = defaultdict(list)
_lock = threading.Lock()


def subscribe(report_id: str) -> queue.Queue:
    q = queue.Queue()
    with _lock:
        _subscribers[report_id].append(q)
    return q


def unsubscribe(report_id: str, q: queue.Queue) -> None:
    with _lock:
        subscribers = _subscribers.get(report_id, [])
        if q in subscribers:
            subscribers.remove(q)
        if not subscribers:
            _subscribers.pop(report_id, None)


def publish(report_id: str, message: str) -> None:
    with _lock:
        subscribers = list(_subscribers.get(report_id, []))
    for q in subscribers:
        q.put(message)
