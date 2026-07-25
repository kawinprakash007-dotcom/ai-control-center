from tools.tool_registry import TOOL_REGISTRY


class CapabilityRegistry:

    def get_executor(self, tool_name: str):

        return TOOL_REGISTRY.get(tool_name)