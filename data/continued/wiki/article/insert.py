from data.continued.wiki.db import WikiArticles, WikiCategories

if __name__ == "__main__":
    fetched_atricles: set[str] = set(WikiArticles.select(WikiArticles.name).where(WikiArticles.stage == "pending"))
    categories = list(WikiCategories.select().where(WikiCategories.status == "pending"))
