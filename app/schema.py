from app.models import Users as UserModel
from app.models import Posts as PostModel

import graphene
from graphene import relay
from graphene_sqlalchemy import SQLAlchemyConnectionField, SQLAlchemyObjectType


class ActiveSQLAlchemyObjectType(SQLAlchemyObjectType):
    class Meta:
        abstract = True
    #
    # @classmethod
    # def get_node(cls, info, id):
    #     return cls.get_query(info).filter(
    #         and_(cls._meta.model.deleted_at==None,
    #              cls._meta.model.id==id)
    #         ).first()


class User(ActiveSQLAlchemyObjectType):
    class Meta:
        model = UserModel
        # interfaces = (relay.Node, )

#
# class Post(ActiveSQLAlchemyObjectType):
#     class Meta:
#         model = PostModel
#         interfaces = (relay.Node, )
#
#
class Query(graphene.ObjectType):
    # node = relay.Node.Field()
    hello = graphene.String()
    users = graphene.List(User)

    def resolve_users(self, info):
        query = User.get_query(info)  # SQLAlchemy query
        return query.all()

    def resolve_hello(self, info):
        return "Hello"
#
#     # Allow only single column sorting
#     # all_users = SQLAlchemyConnectionField(User.connection)
#     # all_employees = SQLAlchemyConnectionField(
#         # Employee.connection, sort=Employee.sort_argument())
#     # Allows sorting over multiple columns, by default over the primary key
#     # all_roles = SQLAlchemyConnectionField(Role.connection)
#     # Disable sorting over this field
#     # all_departments = SQLAlchemyConnectionField(Department.connection, sort=None)
#
#
schema = graphene.Schema(query=Query)