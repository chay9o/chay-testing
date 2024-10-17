from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),

    # ---------- Vector Management ----------

    path('generate_ppt/', views.generate_ppt, name='generate_ppt'),
    path('get_task_status/<str:task_id>/', views.get_task_status, name='get_task_status'),
    path('get_task_status1/<str:task_id>/', views.get_task_status1, name='get_task_status1'),
    path('download_ppt/<str:task_id>/', views.download_ppt, name='download_ppt'),
    path('download_ppt1/<str:task_id>/', views.download_ppt1, name='download_ppt1'),
    path('evaluate_text', views.evaluate_text, name='evaluate_text'),
    path('stinsight_step1', views.stinsight_step1, name='stinsight_step1'),
    path('stinsight_step2', views.stinsight_step2, name='stinsight_step2'),
    path('stinsight_step3', views.stinsight_step3, name='stinsight_step3'),
    path('stinsight_step4', views.stinsight_step4, name='stinsight_step4'),
    path('stinsight_step5', views.stinsight_step5, name='stinsight_step5'),
    path('stinsight_step6', views.stinsight_step6, name='stinsight_step6'),
    path('DA_tester', views.DA_tester, name='DA_tester'),
    path('generate_questions', views.generate_questions, name='generate_questions'),

    path('create-solution/', views.create_solution, name='create_solution'),
    path('delete-solutions/', views.delete_solutions, name='delete_solutions'),



    path('create-collection/', views.create_collection),
    path('add-object/text/', views.add_vectors_text),
    path('add-object/file/', views.add_vectors_file),
    path('add-object/all/', views.add_vectors_all),
    path('chat-summary/', views.chat_summary),

    path('update-object/text/', views.update_chunk_text),
    path('update-object/file/', views.update_chunk_file),
    path('update-object/all/', views.update_chunk_all),

    path('remove-object/', views.remove_object),
    path('remove-objects/entity/', views.remove_objects_entity),
    path('remove-objects/uuid/', views.remove_objects_uuid),
    path('remove-objects/uuids/', views.remove_objects_uuids),
    path('remove-collection/', views.remove_collection),
    path('remove-objects/file/', views.remove_objects_uuid_file),
    path('remove-objects/all/', views.remove_objects_all),
    path('remove-objects/private-user/', views.remove_user_objects),

    path('get-object/', views.get_object),
    path('get-objects/entity/', views.get_objects_entity),
    path('get-objects/uuid/', views.get_objects_uuid),
    path('get-collection/', views.get_collection),

    path('search-hybrid/', views.search_hybrid),
    # ---------- ---------- ---------- ---------- ----------

    # ---------- Master Vectors ----------
    path('create-master-collection/', views.create_master_collection),
    path('add-master-object/text/', views.add_master_vectors),
    path('add-master-object/file/', views.upload_master_file),

    path('remove-master-object/', views.remove_master_object),
    path('remove-master-objects/filename/', views.remove_master_objects_file),
    path('remove-master-objects/uuid/', views.remove_master_objects_uuid),
    path('remove-master-collection/', views.remove_master_collection),

    path('get-master-object/', views.get_master_object),
    path('get-master-objects/filename/', views.get_master_objects_filename),
    path('get-master-objects/uuid/', views.get_master_objects_uuid),
    path('get-master-objects/type/', views.get_master_objects_type),
    path('get-master-collection/', views.get_master_collection),
    # ---------- ---------- ---------- ---------- ----------

    # ---- CALL THIS WHEN LIMIT EXCEEDS!!!!!
    path('destroy/', views.destroy_all),
    # ---------- ---------- ---------- ---------- ----------

    # ---------- BACKUPS ----------
    # MASTERS VECTORS
    path('backup-master-collection/', views.backup_master_vectors),
    path('restore-master-collection/', views.restore_master_vectors),

    # COMPANY VECTORS
    path('backup-collection/', views.backup_company_vectors),
    path('restore-collection/', views.restore_company_vectors),
    # ---------- ---------- ---------- ---------- ----------

]
