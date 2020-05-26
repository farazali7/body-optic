from flask import Flask
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config["SQL_ALCHEMY_DATABASE_URI"] = 'sqlite:///database.db'

    from .view import main
    app.register_blueprint(main)


    return app