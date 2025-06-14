from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('proje.html')

@app.route('/num1_hour')
def num1_hour():
    return render_template('num1_hour.html')

@app.route('/num1_day')
def num1_day():
    return render_template('num1_day.html')

@app.route('/num2_hour')
def num2_hour():
    return render_template('num2_hour.html')

@app.route('/num2_day')
def num2_day():
    return render_template('num2_day.html')

@app.route('/num4_hour')
def num4_hour():
    return render_template('num4_hour.html')

@app.route('/num4_day')
def num4_day():
    return render_template('num4_day.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
