from flask import Flask, request
from app import views
app = Flask(__name__)

app.add_url_rule(rule="/",endpoint="home",view_func=views.index)
app.add_url_rule(rule="/app",endpoint="app",view_func=views.app)
app.add_url_rule(rule="/app/gender/",endpoint="gender",view_func=views.gender,methods=['GET', 'POST'])
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

