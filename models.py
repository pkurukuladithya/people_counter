from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class CountEvent(db.Model):
    __tablename__ = "count_events"

    id = db.Column(db.Integer, primary_key=True)
    direction = db.Column(db.String(10))  # "IN" or "OUT"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    lobby_count = db.Column(db.Integer)   # people currently in lobby after this event
