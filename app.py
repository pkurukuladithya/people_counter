from flask import Flask, render_template, Response
from config import Config
from models import db, CountEvent
from counter import PeopleCounter

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Initialize database and YOLO counter
with app.app_context():
    db.create_all()
    people_counter = PeopleCounter(camera_index=0)  # 0 = default webcam


@app.route("/")
def index():
    # Compute totals from DB
    total_in = db.session.query(CountEvent).filter_by(direction="IN").count()
    total_out = db.session.query(CountEvent).filter_by(direction="OUT").count()
    # latest event defines lobby_count, else 0
    last_event = CountEvent.query.order_by(CountEvent.timestamp.desc()).first()
    lobby_count = last_event.lobby_count if last_event else 0

    # also sync in counter object (optional safety)
    people_counter.in_count = total_in
    people_counter.out_count = total_out
    people_counter.lobby_count = lobby_count

    return render_template(
        "index.html",
        total_in=total_in,
        total_out=total_out,
        lobby_count=lobby_count
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        people_counter.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/history")
def history():
    events = CountEvent.query.order_by(CountEvent.timestamp.desc()).limit(200).all()
    return render_template("history.html", events=events)


if __name__ == "__main__":
    app.run(debug=True)
