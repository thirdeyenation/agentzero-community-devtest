from dataclasses import dataclass, field
import json
from typing import Optional, Dict
import uuid


@dataclass
class LogItem:
    log: 'Log'
    no: int
    type: str
    heading: str
    content: str
    kvps: Optional[Dict] = None
    guid: str = ""

    def __post_init__(self):
        self.guid = self.log.guid

    def update(self, type: str | None = None, heading: str | None = None, content: str | None = None, kvps: dict | None = None):
        if self.guid == self.log.guid:
            self.log.update_item(self.no, type=type, heading=heading, content=content, kvps=kvps)

    def output(self):
        return {
            "no": self.no,
            "type": self.type,
            "heading": self.heading,
            "content": self.content,
            "kvps": self.kvps
        }

class Log:

    def __init__(self):
        self.guid: str = str(uuid.uuid4())
        self.updates: list[int] = []
        self.logs: list[LogItem] = []

    def log(self, type: str, heading: str | None = None, content: str | None = None, kvps: dict | None = None) -> LogItem:
        item = LogItem(log=self,no=len(self.logs), type=type, heading=heading or "", content=content or "", kvps=kvps)
        self.logs.append(item)
        self.updates += [item.no]
        return item

    def update_item(self, no: int, type: str | None = None, heading: str | None = None, content: str | None = None, kvps: dict | None = None):
        item = self.logs[no]
        if type is not None:
            item.type = type
        if heading is not None:
            item.heading = heading
        if content is not None:
            item.content = content
        if kvps is not None:
            item.kvps = kvps
        self.updates += [item.no]

    def output(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.updates)
        
        out = []
        seen = set()
        for update in self.updates[start:end]:
            if update not in seen:
                out.append(self.logs[update].output())
                seen.add(update)
        
        return out
               


    def reset(self):
        self.guid = str(uuid.uuid4())
        self.updates = []
        self.logs = []
