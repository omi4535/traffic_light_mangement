import pickle
from flask import Flask, render_template, request  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
from datetime import datetime


app = Flask(__name__)

app = Flask(__name__, static_url_path="/static", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


model = pickle.load(open("model.pkl", "rb"))


@app.route("/predict", methods=["POST"])  # Assuming this route is for POST requests
def predict():
    features = [x for x in request.form.values()]

    # Parse the input string into a datetime object

    date_obj = datetime.datetime.strptime(features[3], "%Y-%m-%d")

    # Format the datetime object as "Month DayOrdinal Year"
    formatted_date = (
        date_obj.strftime("%B %dst %Y")
        .replace(" 1st", " 1st")
        .replace(" 2nd", " 2nd")
        .replace(" 3rd", " 3rd")
    )
    # Ensure that form data contains at least two values
    if len(features) < 2:
        return (
            "Insufficient data provided",
            400,
        )  # Returning an error response if insufficient data

    # Check if the second value is "junction1"
    if features[2] == "mg":
        junction = 1
        chowk = "M.G.Road"
    elif features[2] == "Chandani":
        junction = 2
        chowk = "Chandani Chowk"
    elif features[2] == "Katraj":
        junction = 3
        chowk = "Katraj Chowk"
    elif features[2] == "mk":
        junction = 4
        chowk = "M.K.Road"

    day = get_day_number(features[3])
    is_holiday_flag = is_holiday(features[3])
    if is_holiday_flag:
        IsHoliyday = 1
        holi = formatted_date + " has a holiday in " + features[0]
    else:
        IsHoliyday = 0
        holi = ""
    hour = get_hour_24_format(features[4])

    if IsHoliyday == 0 and (day == 5 or day == 6):
        IsHoliyday = 1
        holi = formatted_date + " has a holiday in " + features[0]

    data_to_predict = pd.DataFrame(
        {
            "Junction": [junction],
            "Hour": [hour],
            "DayOfWeek": [day],
            "IsWeekend": [IsHoliyday],
        }
    )
    prediction = model.predict(data_to_predict)
    # Returning extracted features if the junction is not "junction1"

    if prediction == 1:
        return render_template(
            "index.html",
            prediction="on "
            + formatted_date
            + ", at "
            + chowk
            + ", the traffic light must be ON at "
            + features[4]
            + ".",
            holiday=holi,
        )
    elif prediction == 0:
        return render_template(
            "index.html",
            prediction="on "
            + formatted_date
            + ", at "
            + chowk
            + ", the traffic light must be OFF at "
            + features[4]
            + ".",
            holiday=holi,
        )


import datetime


def get_day_number(date):
    # Parse the input date string into a datetime object
    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")

    # Get the weekday number (0 for Monday, 1 for Tuesday, ..., 6 for Sunday)
    weekday_number = date_obj.weekday()

    # Convert the weekday number to the desired range (1 to 7)
    day_number = (weekday_number + 1) % 7 + 1

    return day_number


import datetime
import holidays


def is_holiday(date_str, country="IND"):
    # Parse the input date string into a datetime object
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    # Get the holidays for the specified country and year
    holiday_list = holidays.CountryHoliday(country, years=date_obj.year)

    # Check if the date is a holiday
    return date_obj in holiday_list


def get_hour_24_format(time_str):
    # Parse the input time string into a datetime object
    time_obj = datetime.datetime.strptime(time_str, "%H:%M")

    # Extract hours
    hour_24_format = time_obj.hour

    return hour_24_format


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
