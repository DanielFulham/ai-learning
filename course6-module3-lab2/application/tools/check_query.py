from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


_CHECKER_SYSTEM_PROMPT = """\
You are a SQL expert with a strong attention to detail.
Double check the SQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are
no mistakes, just reproduce the original query.

Output the final SQL query only.\
"""


def make_check_query(llm: BaseChatModel) -> BaseTool:
    @tool("sql_db_query_checker")
    def sql_db_query_checker(query: str) -> str:
        """Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!"""
        messages = [
            SystemMessage(content=_CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        return str(response.content)

    return sql_db_query_checker