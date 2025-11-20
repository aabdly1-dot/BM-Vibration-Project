# CM2015: Vibration Exposure App - Project Plan

This document outlines the project plan, team roles, and timeline for the CM2015 project (BM-Vibration).

## 👥 Project Teams

* **Team A (App & Integration):** Giovanni & Mustafa
    * **Focus:** iOS app development, Bluetooth sensor connectivity, user interface (UI), and integrating Team B's algorithm into a functional, real-world app.

* **Team B (Signal & Algorithm):** Ibrahim & Roman
    * **Focus:** Signal analysis, filter development, and building the classification algorithms (Classifier v1: Time, Classifier v2: Tool Type). This team is responsible for solving the "just include the signal of vibration" problem.

## 🗓️ Overall Project Timeline

* **Project Start:** October 27, 2025
* **Final Report Deadline:** January 13, 2026
* **Final Presentation:** January 9, 2026

---

## Iteration 1 (Sprint 1): Setup, Data Collection & Initial Prototyping
**(Duration: Oct 27 - Nov 16)**

The goal of this iteration is to get all tools working, collect initial data, and prepare for Seminar 1.

**🔴 Key Deadlines:**
* **Nov 7:** Ethics Seminar submission
* **Nov 11:** Ethics Seminar
* **Nov 12:** **Seminar 1 (Compulsory)**

### Phase 1.1: Project Kickoff & Setup (All Students)
* [ ] **Kickoff:** Align on project goals and this plan.
* [ ] **Tools:** Set up the shared GitHub repository.
* [ ] **Hardware:** Receive and test all equipment: Movesense IMU sensors, Apple Computer, and the three vibrating tools (the dataset).

### Phase 1.2: Research & Ethics (All Students)
* [ ] **Team B (Signal - Ibrahim & Roman):** Read the pilot study and previous PCC report to understand the "transfer function" (grip vs. back of hand) and existing algorithms.
* [ ] **Team A (App - Giovanni & Mustafa):** Contribute to the Ethics submission (Nov 7) by describing the app and how it will handle user data.
* [ ] **Team B (Signal - Ibrahim & Roman):** Contribute to the Ethics submission (Nov 7) by describing *what* data you will collect and *how* you will test it on a person.
* [ ] **All:** Attend Ethics Seminar (Nov 11).

### Phase 1.3: Initial Prototyping (Team A - App - Giovanni & Mustafa)
* [ ] **Setup:** Set up the Apple Computer with the iOS development environment (Xcode).
* [ ] **Milestone 1 (App):** Build a basic "Hello World" app that successfully connects to the Movesense sensor via Bluetooth.
* [ ] **Milestone 2 (App):** Make the app display the raw X, Y, Z accelerometer data streaming from the sensor to the phone's screen.

### Phase 1.4: Data Collection (Team B - Signal - Ibrahim & Roman)
* [ ] **Data Collection:** Use the standard Movesense app to collect your first datasets.
* [ ] **"Signal" Data:** Record 5-10 minutes of data for each of the 3 vibrating tools.
* [ ] **"Noise" Data:** This is critical. Record data while walking, walking up stairs, tapping a table, and swinging your arms. This is the data you need to filter out.

### Phase 1.5: Seminar 1 (All Students)
* [ ] **Prepare & Present:** Prepare slides and present your progress at Seminar 1 (Nov 12).
    * **Team A (Giovanni & Mustafa):** Show the working "Hello World" app and discuss the 8-hour measurement challenge.
    * **Team B (Ibrahim & Roman):** Show your data collection plan and the challenges of separating "noise" from "signal."

---

## Iteration 2 (Sprint 2): Core Algorithm & App Skeleton
**(Duration: Nov 17 - Dec 14)**

This is the main development phase. Both teams work in parallel on the core components.

**🔴 Key Deadlines:**
* **Dec 11:** **Seminar 2 (Compulsory)**

### Phase 2.1: Algorithm Development (Team B - Signal - Ibrahim & Roman)
* [ ] **Task 1: The Filter:** This is your main goal. Using the "Noise" data from Sprint 1, develop and validate a signal processing filter (e.g., a high-pass filter) that removes all non-tool movements (walking, etc.).
* [ ] **Task 2: Classifier v1 (Time):** Build the primary algorithm (in Python or a similar tool). Its only job is to take the *filtered* signal and output "Vibration ON" or "Vibration OFF".
* [ ] **Task 3: Classifier v2 (Tool Type):** Begin developing the secondary algorithm. When "Vibration ON" is detected, this algorithm analyzes the signal (using FFTs) to try and identify *which* of the 3 tools is being used.

### Phase 2.2: App Development (Team A - App - Giovanni & Mustafa)
* [ ] **Task 1: App UI:** Design and build the main app interface. It needs a "Connect" screen and a "Dashboard" to show the results.
* [ ] **Task 2: The 8-Hour Challenge:** This is your main goal. Research and implement a solution to keep the app running in the background for 8 hours. This includes:
    * Managing battery life and app sleep (e.g., investigating "Guided Access").
    * Handling Bluetooth data loss and reconnections.
* [ ] **Task 3: Integration (Simple):** Create a placeholder in the app to receive a simple "ON/OFF" signal and use it to start/stop a timer.

### Phase 2.3: Seminar 2 (All Students)
* [ ] **Prepare & Present:** Prepare and present demos of your progress at Seminar 2 (Dec 11).
    * **Team A (Giovanni & Mustafa):** Demo the app UI and your solution for the 8-hour challenge.
    * **Team B (Ibrahim & Roman):** Demo your algorithms working on the test data.

---

## Iteration 3 (Sprint 3): Integration & Validation
**(Duration: Dec 15 - Jan 4)**

This sprint is focused on merging the two parts and preparing all materials for the final deadlines. This sprint includes the holiday break.

**🔴 Key Deadlines:**
* **Dec 18:** **Deadline 1 Report (First Draft)**
* **Jan 2:** **Deadline Peer Review Feedback**

### Phase 3.1: Code Integration (All Students)
* [ ] **Integration:** Team B (Ibrahim & Roman) must provide their final, working algorithms (Classifier v1 & v2) to Team A.
* [ ] **Integration:** Team A (Giovanni & Mustafa) must integrate these into the live iOS app.

### Phase 3.2: Validation (The "Gold Standard") (All Students)
* [ ] **Perform Tests:** Perform the full validation as a team. One student uses the tool, another uses a stopwatch.
* [ ] **Record Results:** Record all results for the "stopwatch vs. app" comparison.
* [ ] **Test Scenarios:** You must test all scenarios: all 3 tools, strong vs. loose grip, glove vs. no glove, and walking/stairs (to test the filter).

### Phase 3.3: Reporting & Review (All Students)
* [ ] **Report 1 (Dec 18):** Write and submit your first full draft of the final report. This *must* include the results from your validation tests.
* [ ] **Peer Review (Jan 2):** Review the reports from other groups and provide constructive feedback. Read and incorporate the feedback you receive into your own final report.

---

## Iteration 4 (Sprint 4): Final Delivery
**(Duration: Jan 5 - Jan 13)**

The final push to complete all course deliverables.

**🔴 Key Deadlines:**
* **Jan 7:** **Poster Deadline (for printing)**
* **Jan 9:** **Final Presentation (Compulsory)**
* **Jan 12:** **Deadline Individual Assignment**
* **Jan 13:** **Deadline Final Report**

### Phase 4.1: Final Deliverables (All Students)
* [ ] **Poster (Jan 7):** Design and submit your final poster based on your project and validation results.
* [ ] **Presentation (Jan 9):** Prepare and practice your final presentation. This should include a *live demo* of the app connecting to the sensor and successfully classifying vibration vs. no vibration.
* [ ] **Final Report (Jan 13):** Submit your final, polished report, incorporating all feedback and final results.
* [ ] **Individual Assignment (Jan 12):** Complete and submit your individual reflections.
