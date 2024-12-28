from rnn_translate import models
import mongoengine as me


def init_admin():
    if models.User.objects(username="admin").first():
        print("already have admin user in database")
        return
    admin_user = models.User(username="admin", password="admin123", roles=["admin"])
    admin_user.set_password("admin123")
    admin_user.save()
    print("Admin user created")
