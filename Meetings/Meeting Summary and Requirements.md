# Chapter 1: The Core Problem & Project Goal

The central problem this project addresses is that current methods for measuring vibration exposure are highly inaccurate. Ergonomists and workers typically "guess" the exposure time, and studies confirm that this self-reported time is "heavily overestimating" the true duration.

The existing ISO standards for vibration dose limits, which were established in the 1980s, are also flawed. They were likely based on asking workers (flawed self-reports) rather than direct observation. This creates a paradox: if we now measure the true (and much shorter) exposure time, it might incorrectly suggest that employers can make people work longer before hitting the official limit.

Therefore, the primary goal of this project is to create a system that can accurately "measure that time" of true vibration exposure.

---

# Chapter 2: Project Priorities (The 3 Goals)

The project has a clear, 3-step priority list for development:

## Priority 1: Measure Exposure Time (On/Off Classifier)
* The "first and most important aim is the timing".
* Your first objective is to build a classifier that takes the sensor signal and robustly decides "yes, there is vibration" or "no, there is not vibration".
* This simple "on/off" classifier addresses the "largest error" in current ergonomic assessments.

## Priority 2: Identify Different Tools (Secondary Classifier)
* Once timing is solved, the next priority is to distinguish between different tools.
* This secondary classifier would analyze the vibration profile (e.g., using clustering or FFTs) to see if it can identify "this is one tool, this is a different tool".

## Priority 3: Measure Amplitude (Tertiary Goal)
* This is the final and lowest-priority goal. The supervisor "would be happy" if this is included, but only after the first two goals are met.
* (This is complex because the official ISO filter is based on subjective feeling rather than objective harm, but the focus for now is on time and tool type).

---

# Chapter 3: Hardware and Core Concept

* **Hardware:** The project will use the Movesense sensor.
* **Software:** The sensor will connect via Bluetooth to a mobile app (likely iOS) that you will build. You will be provided with the Movesense app's open-source code as a starting point.
* **Core Concept (Sensor Placement):** Standard measurements are taken from the inside (gripping) part of the hand. Your project will place the sensor on the "outside of the hand". A previous project has already studied the "transfer function" to correlate the data from these two placements, so you can build on their work.

---

# Chapter 4: The Validation Plan (The "Gold Standard")

To prove your system works, you must test it against a "gold standard".

* **Gold Standard Defined:** You will compare your app's results against reality by using a stopwatch to time the true power-on state of a tool.
* **Required Test Cases:** Your validation protocol must include:
    * Testing with at least three different tools (e.g., drill, grinder).
    * Testing the difference between a "strong grip" and a "loose grip".
    * Testing "with a glove" and "without a glove".
    * Testing "negative data" to ensure your filter works. This means recording data while "stair walking," running, or jumping to confirm the app does not log this as vibration time.

---

# Chapter 5: Team Skills and Project Management

* **Team Skills Assessment:**
    * **Strengths:** Signal processing (FFTs, filters), Python, and some C++ (Arduino).
    * **Challenges:** App development (no prior iOS/Android experience), Bluetooth connectivity, and firmware coding.
* **Key Advice:**
    * **Time Management:** Do not "hyper-focus" on signal processing just because it's a team strength. You must "allocate time appropriately" for the more challenging app development side to ensure you have a complete, working product at the end.
    * **Documentation:** "Work with version control... on everything," not just your code. This includes all your documentation, requirements, and assumption lists, using tools like GitHub or Google Docs.

---

# Chapter 6: Technical Challenge 1 - Filtering Noise

This chapter explains the main technical problem: how to separate tool vibration from movement "noise."

* **The Problem:** Raw accelerometer data shows a constant value (e.g., 9.8 m/s²) even when resting, which is gravity. More importantly, basic movements like walking ("going around") or even having the sensor in a pocket can be misread as tool vibration by simpler systems. A competitor's watch was noted to have this exact flaw.
* **The Solution:** The key is to use a high-pass filter. This filter removes the "slow variation" signals (like walking or the constant pull of gravity), making the signal "very flat" (near zero) when no high-frequency tool vibration is present. This is the "trick" to isolating only the tool's signal.

---

# Chapter 7: Technical Challenge 2 - 8-Hour Measurements & Data Loss

A key requirement is to measure for a full "eight-hour" workday. This presents two major technical challenges:

* **App Backgrounding:** iOS and Android will automatically "kill apps" running in the background to save battery. This was a problem discovered in previous projects. A potential workaround is using the "Guided Access" feature on iOS, which locks the phone onto a single app and prevents it from sleeping.
* **Bluetooth Data Loss:** Over a long period, you will experience "dropped packages" (lost data). This can be caused by low battery, distance, or other environmental "signal saturation". If you have a 15% drop rate, your 8-hour recording will be missing over an hour of data. Your software must have a strategy for "accounting for signal loss" and "re-synchronizing" the data to get an accurate total time.

---

# Chapter 8: Technical Challenge 3 - Firmware vs. App Processing

A key "trade-off" you must consider is where to do the signal processing.

1.  **Option 1 (App Processing):** Stream all raw, high-frequency data from the sensor to the phone and have the app do the filtering. This is simple to code but has a high risk of data loss and drains the phone's battery.
2.  **Option 2 (Firmware Processing):** Program the firmware of the Movesense sensor itself.

* **The "Vision" (Recommended):** The ideal solution is for the sensor to sample at a high frequency internally, perform all the filtering internally, and then send a simple, processed signal to the app (e.g., once per second) that just says "Vibration: ON, Amplitude: 2.3". This would solve the data loss and battery issues.

---

# Chapter 9: Final Advice and Next Steps

* **Project Codebase:** You should NOT use the old "ErgoHandMeter" project code as a base. It was noted that previous students "gave up within two weeks" because the code is "not readable". You should start with the official Movesense iOS app examples, which are "way more well-documented and cleaner code".
* **Immediate Next Steps:**
    1.  The team should meet to create a time plan and, most importantly, write down a list of project "requirements and assumptions" (e.g., "We assume the user is not running," "We are only measuring hand-arm vibration").
    2.  You will present this plan and your assumptions to the supervisors for discussion before proceeding.
    3.  The supervisor will send you the conference presentation that includes the high-pass filter design.
    4.  You will be given the Movesense sensor and access to tools for testing.
