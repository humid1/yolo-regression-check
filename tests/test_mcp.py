import shutil
import sys
import tempfile
import unittest
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(find_spec("mcp"), "Install the MCP extra to run this integration test")
class McpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_comparison_and_workspace_boundaries(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            workspace = Path(directory) / "workspace"
            shutil.copytree(PROJECT / "demo", workspace / "demo")
            shutil.copyfile(PROJECT / "demo" / "candidate.json", workspace.parent / "outside.json")
            server = StdioServerParameters(
                command=sys.executable,
                args=["-B", "-m", "yolo_regression.mcp_server", "--workspace", str(workspace)],
                cwd=PROJECT,
            )
            async with stdio_client(server) as (reader, writer), ClientSession(
                reader, writer, read_timeout_seconds=timedelta(seconds=15),
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.assertEqual([tool.name for tool in tools.tools], ["compare_yolo_predictions"])
                tool_name = tools.tools[0].name
                self.assertIsNotNone(tools.tools[0].outputSchema)
                args = {"annotations": "demo/annotations.json", "baseline": "demo/baseline.json",
                        "candidate": str(workspace / "demo" / "candidate.json"), "max_recall_drop_pp": 5,
                        "max_new_misses": 1, "gate_class_ids": [0]}
                result = await session.call_tool(tool_name, args)
                self.assertFalse(result.isError, result.content)
                summary = result.structuredContent
                self.assertEqual(summary["gate_status"], "failed")
                self.assertEqual(summary["failed_category_ids"], [0])
                self.assertEqual([failure["metric"] for failure in summary["gate_failures"]],
                                 ["recall_drop_pp", "new_misses"])
                self.assertEqual(summary["class_changes"][0]["recall_drop_pp"], 25)
                self.assertEqual(summary["new_misses"], 2)
                self.assertFalse(summary["provenance"]["coverage_verified"])
                self.assertTrue(summary["sample_previews"][0]["gate_evidence"])
                self.assertEqual(len(summary["findings"]), 2)

                report = Path(summary["report_path"])
                self.assertEqual(report.parent, workspace / "reports")
                self.assertIn("75.0% → 50.0%", report.read_text(encoding="utf-8"))
                original = report.read_bytes()
                repeated = await session.call_tool(tool_name, {**args, "output_html": str(report)})
                self.assertTrue(repeated.isError, repeated.content)
                self.assertEqual(report.read_bytes(), original)

                for arguments in ({**args, "candidate": "../outside.json"},
                                  {**args, "baseline_manifest": "../outside.json"},
                                  {**args, "output_html": "../outside.html"}):
                    rejected = await session.call_tool(tool_name, arguments)
                    self.assertTrue(rejected.isError, rejected.content)
                    self.assertIn("configured workspace", rejected.content[0].text)
                self.assertFalse((workspace.parent / "outside.html").exists())
