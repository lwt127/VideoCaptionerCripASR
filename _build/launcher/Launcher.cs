using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

// VideoCaptioner double-click launcher.
//
// This tiny executable lives in the project root (next to main.py) and starts
// the application using the project's own virtual environment, with no console
// window. All relative paths (resource/, AppData/, work-dir/) and runtime
// downloads keep working because the working directory is set to the project
// root, exactly as if you had run `python main.py` there.
static class Launcher
{
    [STAThread]
    static int Main(string[] args)
    {
        // Folder where this .exe lives = project root.
        string exePath = Assembly.GetExecutingAssembly().Location;
        string root = Path.GetDirectoryName(exePath);

        string pythonw = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
        string python  = Path.Combine(root, ".venv", "Scripts", "python.exe");
        string mainPy  = Path.Combine(root, "main.py");

        // Prefer the windowed interpreter (no console). Fall back to python.exe.
        string interpreter = File.Exists(pythonw) ? pythonw : python;

        if (!File.Exists(interpreter))
        {
            MessageBox.Show(
                "找不到虚拟环境 Python 解释器：\n" + pythonw +
                "\n\n请先在项目目录创建虚拟环境并安装依赖：\n" +
                "  py -3.13 -m venv .venv\n" +
                "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
                "VideoCaptioner 启动失败",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        if (!File.Exists(mainPy))
        {
            MessageBox.Show(
                "找不到 main.py：\n" + mainPy,
                "VideoCaptioner 启动失败",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        // Pass through any extra command-line arguments.
        string extra = "";
        foreach (string a in args)
            extra += " \"" + a + "\"";

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = interpreter,
                Arguments = "\"" + mainPy + "\"" + extra,
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "启动 VideoCaptioner 时发生错误：\n\n" + ex.Message,
                "VideoCaptioner 启动失败",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
