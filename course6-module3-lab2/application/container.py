from application.sql_agent import SqlAgent
from application.tools.check_query import make_check_query
from application.tools.get_schema import make_get_schema
from application.tools.list_tables import make_list_tables
from application.tools.run_query import make_run_query
from domain.models import AgentTrace
from infra.openai_chat_model import OpenAIChatModelProvider
from infra.sqlalchemy_database import SqlAlchemyDatabase
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.sql_database_interface import SqlDatabaseInterface


def initialise(
    database_uri: str,
    chat_model_provider: ChatModelProviderInterface | None = None,
    sql_database: SqlDatabaseInterface | None = None,
) -> SqlAgent:
    if chat_model_provider is None:
        chat_model_provider = OpenAIChatModelProvider()
    if sql_database is None:
        sql_database = SqlAlchemyDatabase(database_uri)

    llm = chat_model_provider.create()
    trace = AgentTrace()

    tools = [
        make_list_tables(sql_database),
        make_get_schema(sql_database),
        make_run_query(sql_database),
        make_check_query(llm),
    ]

    return SqlAgent(
        llm=llm,
        tools=tools,
        dialect=sql_database.dialect,
        trace=trace,
    )