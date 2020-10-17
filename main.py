import json
import graphene
import uuid
from pprint import pprint

class User(graphene.ObjectType):
    id = graphene.ID()
    username = graphene.String()
    n_posts = graphene.Int(required=False)


users = [
    User(id=uuid.uuid4(), username="Jack", n_posts=0),
    User(id=uuid.uuid4(), username="Bob", n_posts=10),
    User(id=uuid.uuid4(), username="Alice", n_posts=100)
]

class Query(graphene.ObjectType):
    hello = graphene.String(name=graphene.String())
    users = graphene.List(User, post_threshold=graphene.Int())

    def resolve_hello(self, info, name):
        return "Hello" + name

    def resolve_users(self, info, post_threshold):
        # print(args)
        return [user for user in users if user.n_posts >= post_threshold]


class CreateUser(graphene.Mutation):

    class Arguments:
        username = graphene.String()

    user = graphene.Field(User)

    def mutate(self, info, username):
        user = User(id=uuid.uuid4(), username=username)
        return CreateUser(user=user)

class Mutations(graphene.ObjectType):
    create_user = CreateUser.Field()

schema = graphene.Schema(query=Query, mutation=Mutations)
result = schema.execute('''
    mutation  {
        createUser(username: "j"){
            user {
                username
                
                }
            }
        }

''')
print(result)