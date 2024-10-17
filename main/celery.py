from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')

app = Celery('main')

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Configuration to use prefork pool and limit concurrency
app.conf.update(
    worker_pool='prefork',
    worker_concurrency=1,  # Limit the number of worker processes
    worker_max_tasks_per_child=1,  # Restart workers after each task to free memory
    task_always_eager=False,  # Disable eager execution
)
