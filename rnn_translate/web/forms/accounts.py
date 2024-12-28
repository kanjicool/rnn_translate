from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField
from flask_mongoengine.wtf import model_form
from rnn_translate.web import models

BaseLoginForm = model_form(
    models.User,
    FlaskForm,
    only=["username", "password"],
)


class LoginForm(BaseLoginForm):
    password = PasswordField("Password")
