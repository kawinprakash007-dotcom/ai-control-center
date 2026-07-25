class ContextEngine:

    def __init__(self):

        self.context = {

            "current_app": None,

            "current_task": None,

            "last_tool": None,

            "last_user_message": None,

            "last_ai_response": None,

            "current_goal": None,

            "conversation_count": 0
        }

    def update_user_message(self, message):

        self.context["last_user_message"] = message

        self.context["conversation_count"] += 1

    def update_ai_response(self, response):

        self.context["last_ai_response"] = response

    def set_current_app(self, app):

        self.context["current_app"] = app

    def set_current_task(self, task):

        self.context["current_task"] = task

    def set_last_tool(self, tool):

        self.context["last_tool"] = tool

    def set_goal(self, goal):

        self.context["current_goal"] = goal

    def get_context(self):

        return self.context