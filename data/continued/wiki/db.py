import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from peewee import CharField, CompositeKey, Model, PostgresqlDatabase

load_dotenv()

url = urlparse(os.environ["DATABASE_URL"])

db = PostgresqlDatabase(
    url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port,
    sslmode="require",
)


class BaseModel(Model):
    class Meta:
        database = db


class WikiCategories(BaseModel):
    name = CharField(max_length=32)
    lang = CharField(max_length=8)
    status = CharField(max_length=8)

    class Meta:
        table_name = "wiki_categories"
        primary_key = CompositeKey("name", "lang")


class WikiArticles(BaseModel):
    name = CharField(max_length=32)
    lang = CharField(max_length=8)
    stage = CharField(max_length=8)

    class Meta:
        table_name = "wiki_articles"
        primary_key = CompositeKey("name", "lang")


class WikiArticleCategories(BaseModel):
    title = CharField(max_length=32)
    lang = CharField(max_length=8)
    category = CharField(max_length=32)

    class Meta:
        table_name = "wiki_articles"
        primary_key = CompositeKey("title", "lang", "category")
