<div align="center">

# 🤖 AI Control Center

### Intelligent • Modular • Local AI Assistant

<p align="center">
An AI-powered desktop assistant that integrates local Large Language Models (LLMs), voice interaction, intelligent planning, and desktop automation.
</p>

![Status](https://img.shields.io/badge/Status-Under%20Development-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

# 📌 Overview

AI Control Center is a modular desktop AI assistant designed to automate everyday computer tasks using local AI models.

Unlike cloud-based assistants, AI Control Center focuses on **privacy**, **speed**, and **offline intelligence** by integrating local LLMs with automation modules.

The project is designed to become a personal AI operating system capable of understanding user commands, planning tasks, and executing them autonomously.

---

# ✨ Features

- 🧠 Local LLM Integration (Ollama)
- 🎙️ Voice Interaction
- ⚡ Desktop Automation
- 🧩 Modular AI Agents
- 💬 Natural Language Processing
- 📂 File Management
- 🌐 Browser Automation
- 🔌 Plugin Support
- 🔒 Privacy Focused
- 🖥️ Cross Platform

---

# 🏗️ System Architecture

```text
                           +----------------------+
                           |        User          |
                           +----------+-----------+
                                      |
                             Voice / Text Input
                                      |
                                      v
                     +-------------------------------+
                     |     AI Control Center UI      |
                     +---------------+---------------+
                                     |
                                     |
                   +-----------------+------------------+
                   |                                    |
                   |                                    |
                   v                                    v
          +-------------------+              +------------------+
          | Voice Processing  |              | Command Parser   |
          +---------+---------+              +---------+--------+
                    |                                  |
                    +----------------+-----------------+
                                     |
                                     v
                         +-------------------------+
                         |      AI Planner         |
                         +------------+------------+
                                      |
                    +-----------------+------------------+
                    |                                    |
                    |                                    |
                    v                                    v
             +-------------+                    +----------------+
             | Local LLM   |                    | Task Executor  |
             | (Ollama)    |                    +-------+--------+
             +------+------+
                    |                               |
                    |                               |
                    +-------------------------------+
                                    |
                                    v
                         +------------------------+
                         |  Operating System API  |
                         +-----------+------------+
                                     |
         ----------------------------------------------------------
        |                |               |              |           |
        v                v               v              v           v
    File System      Applications    Browser       Terminal     Automation
```

---

# ⚙️ Workflow

```text
User
 │
 ▼
Voice/Text Command
 │
 ▼
Speech Recognition
 │
 ▼
Command Parser
 │
 ▼
AI Planner
 │
 ▼
Local LLM (Ollama)
 │
 ▼
Task Executor
 │
 ▼
Operating System
 │
 ▼
Response to User
```

---

# 📂 Project Structure

```text
AI-control-center/
│
├── backend/
│   ├── api/
│   ├── brain/
│   ├── executor/
│   ├── memory/
│   ├── planner/
│   ├── models/
│   └── main.py
│
├── frontend/
│
├── docs/
│   ├── banner.png
│   ├── architecture.png
│   └── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Backend | FastAPI |
| AI Models | Ollama |
| Machine Learning | Transformers |
| Voice Processing | SpeechRecognition |
| Automation | Python |
| Version Control | Git |
| Operating System | Linux / Windows |

---

# 🚀 Current Progress

- ✅ Project Structure
- ✅ Backend Initialization
- ✅ Modular Architecture
- ⏳ Voice Assistant
- ⏳ Local LLM Integration
- ⏳ Task Automation
- ⏳ Plugin System
- ⏳ Desktop GUI

---

# 🗺️ Roadmap

## Phase 1
- Backend Architecture
- AI Brain
- Planner
- Executor

## Phase 2
- Voice Assistant
- Local LLM Integration
- Memory System

## Phase 3
- Desktop Automation
- Browser Automation
- File Management

## Phase 4
- GUI Dashboard
- Plugin Marketplace
- Cross Platform Support

---

# 📸 Screenshots

Project screenshots and demonstration GIFs will be added as development progresses.

```
docs/screenshots/
```

---

# 🚀 Future Improvements

- Voice-controlled desktop assistant
- Multi-agent collaboration
- Computer Vision integration
- Raspberry Pi integration
- Remote control support
- Plugin ecosystem
- Smart scheduling
- Email automation
- Calendar integration
- Android companion app

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/kawinprakash007-dotcom/AI-control-center.git
```

Move into the project

```bash
cd AI-control-center
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the backend

```bash
python backend/main.py
```

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to contribute:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📄 License

This project will be released under the MIT License.

---

# 👨‍💻 Author

## Kawin Prakash

Artificial Intelligence Student

- 🤖 Artificial Intelligence
- 🧠 Machine Learning
- 📡 IoT
- 🐧 Linux
- 💻 Python

GitHub:
https://github.com/kawinprakash007-dotcom

LinkedIn:
https://www.linkedin.com/in/kawin-prakash-68987436b/

---

<div align="center">

⭐ If you like this project, consider giving it a star!

</div>
