release: python manage.py migrate
web: daphne -b 0.0.0.0 -p $PORT main.asgi:application
worker: celery -A main worker --loglevel=info

