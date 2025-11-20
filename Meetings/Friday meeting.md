# The Summay of fridag meeting and the whole transcripte

Chapter 1: Signal Processing & Vibration Detection
The group discussed methods for analyzing accelerometer data to distinguish between human motion (like walking) and tool vibration.

Analysis Techniques: The supervisor recommended looking at the raw signal and applying a Fast Fourier Transform (FFT) to identify specific sub-frequencies. The team should look for high amplitudes in new frequency ranges that appear during tool exposure compared to a baseline.

Frequency Domain: Human movement is generally distinct from the chaotic signals and specific frequencies introduced by vibrating tools.

Baselines: It is important to record a baseline (e.g., 10 minutes of wearing the sensor without tool usage) to know what to compare the vibration data against.

Synthetic Data: If real-world vibration data is scarce, the team can take raw accelerometer data (non-vibrating) and digitally introduce vibration "noise" to test their classification algorithms.

Chapter 2: App Functionality & Requirements
The supervisor outlined specific features the mobile application (iPhone) requires for both development testing and the final user experience.

Real-Time Feedback: The app must plot live data (either the raw signal, max amplitude, or energy) as it comes in. This is crucial for verifying that the sensor is working correctly during measurements.

The "Marker" Feature: For algorithm testing, the app needs a "marker" button. An observer should press this when the user starts and stops using a tool. This creates a "gold standard" to compare against the algorithm's detection.

User Interface: The final output for the user (e.g., a physiotherapist) should display a timeline showing when vibration occurred and calculate the total exposure time.

Chapter 3: Hardware & Tool Selection
The discussion shifted to the physical tools used for testing and the sensors required.

Sensor Hardware: The team is using Movesense sensors and has acquired the necessary laptops. An iPhone is required for the app.

Selecting Test Tools: The supervisor advised against randomly selecting tools. Instead, the team should look at ISO standards and literature to identify which vibration frequencies cause health issues. They should then select test tools (e.g., angle grinders, impact hammers) that match those specific dangerous profiles.

Open Science: If the team collects a high-quality dataset, they should consider publishing it on GitHub, as vibration datasets are currently difficult to find.

Chapter 4: Project Management & Methodology
The supervisor emphasized the need for structured decision-making documentation for the next phase of the project.

Defining the Scope: The team needs to clearly document their requirements, assumptions, and the main trade-offs they face.

Trade-off Tables: When making technical decisions (e.g., choosing an algorithm), the team should use trade-off tables. These tables should list alternatives against criteria (like complexity, cost, or accuracy) to justify why a specific solution was chosen.

Next Steps: The goal for the next meeting is to present a first version of these requirements and trade-off decisions.









# transcripte


"Uh, what have you been working on? What's your idea of moving ahead?

Yes, but.

So we were...

Okay, begin please.

Yes.

So we was like planning on how we will do the job. And uh it was a bit like difficult to to know how to identify or like classify.

Okay.

so that we know like who or it's only like walking or it's a bit unclear until now.

In terms of the signal processing, is what you mean.

Yeah.

Like when you have the signal of the acceleration, what kind of variables can you look at in order to figure out if this is vibration or not?

Exactly.

Okay. Uh, because this is not my specialty, I won't go now into details on this, but I will collect information to send your way and talk with Mikke on sending very specific. Give me a second to note it down. Uh, PCC one. Uh, identify vibrations.

Uh, but in general, uh, you have a series of ways, depending on what type of vibration you're expecting, that you could look at. So, the first thing is that you could look in general at the signal and try to see if you have specific sub-frequencies. So after doing uh a fast Fourier transformation and getting the basic frequencies of a signal, you can go and see at the amplitudes of those sub-frequencies. And then you could have kind of a baseline, and then you see all of a sudden that when you have exposure to a specific tool that is vibrating, you have a new one that comes in with a very high al- uh, amplitude. If that makes sense.

Uh, also you could look just on the frequency domain, and you would see that all of a sudden you have more chaotic signals with a lot of frequencies playing in. And all of a sudden when you have exposure to vibration, everything becomes almost this one.

Uh, you could do techniques that would work also from having a baseline. So, having 10 minutes just wearing the sensor doing things, and then you know what to compare against. So there is a few things that you could do, depending on how you would like to approach it.

Um, but I will note down to send you a few techniques uh on how to do this. From raw accelerometer data.

Um. Yeah. Yeah. Yeah.

And then the app is it like only for uh... (muffled) You're back.

Sorry, you cut out a bit.

Yeah. Yes. So the app, is it still like we only need having how much time the work have like uh used the tool or we like need to know something else?

Sorry, go again.

Yeah, for the app. Because we need also an app in uh an iPhone. So it's only like uh we need like to show the time there or we need like other like information to be showing.

Uh, for the app, there is one side which I think would be beneficial for development and for making sure that it's working. So I think while you're connecting the... connected to the sensor and doing a measurement, you need to have something that plots the actual data, like as it's coming in, because then like... you need to be able to figure out if it's working. You need feedback.

So when you have a vibration, the app needs to show like either if you have a metric for vibration, so this could be like the max amplitude or the energy that is available on the sensor, or you could have just the raw signal being plotted uh as it's coming in. But you would need to have something. Because even when you will be testing it, uh you might have the time output at the end, but if it's not correct and you don't know what happened during the measurement, it will make your life a bit more difficult.

So in general, you would need some features like plotting the live signal as it's coming in. You would also need to have like kind of a start/stop recording. Uh, and then for your work, especially if you end up comparing an algorithm, it would be good to have the ability to have markers. So what this means is that... let's say that you have the recording and the recording file is like a CSV with a timestamp, the X uh vector, Y vector, Z vector. Then you have uh an extra column called marker, and there you could go during the measurements that you're doing in the phone and click on the marker. And this would allow you to say, okay, for example, if you're testing how to detect one tool, you're using... somebody has the tool in their hand. When they start using it, on the phone, the observer presses the marker, so you have a point where the vibration starts. And when they stop using it, you press the marker again. So you have kind of your gold standard, because you have the measurement of like, okay, they used it from here to here, and you have that paired with the actual signal that you're recording. Does that make sense?

Yeah.

Yeah? Good. Uh, so you would have this. And then at the end, of course, you would have probably a time plot, and like uh bars for when you detected vibration.

Uh, that would be the end result, and then a total time. Like the thing that you would use as an ergonomist or as a physiotherapist or an occupational therapist would be that. Like the plot at the end that says like work day, let's say, or work task, like 20 minutes or 8 hours, and like you would have bars for when you have vibration. Down the line, we could go and say, okay, different color codes for different tools, if we get to that stage, or we could do the tools and then if we include amplitude, which we could also include uh x-axis and say, okay, this is higher and it's with tool B. Uh, but besides that, I think for now the aim is like a plot at the end that says when there is vibration, and then a total under it.

Uh, yeah. Yeah.

Unless, of course, sorry. This is one direction to go. If they say... so, and I'm suggesting things to solve problems that I think about when we're when you would do development. If similar problems or you don't prioritize those problems and think that something else is more important for the app, bring it back and we can discuss it. Because now you're like... I I have an mental image of this in my mind and I'm kind of projecting that. But if you figure out a different way to solve this, or that this might need to be prioritized and this might not, you can also get back to me and say, "Okay, I don't think this makes sense. I think this makes sense," and so forth.

Okay, okay.

Okay. So, I think I knew, but the week was very busy. An email from you about vibrational data sets, correct?

Yeah, exactly. I think that that is like the main problem right now.

Okay. Uh, this, what I meant, was that there should be available online. So, open repositories, GitHub, and stuff like that. I'll try to look for stuff as well. Alternatively, I think from the... either you might be able to... for if you're just doing algorithms, you might be able to generate signals.

Uh, there is, for sure, accelerometer data, maybe without vibrations. Like, but there is raw acceleration signals.

So you might be able to do... Could you do that? Yeah, you might be able to do that. Like if you want at least for a software uh, like as you're building up to do a software exercise kind of on this, you could have raw accelerometer data, introduce uh digitally a vibration into the signal, and then work with that for the classifier. So you have a lot of raw non-vibrational data, have some synthetic data with... acceler... with vibrations included, and then see if your classifier can pick it up. If that makes sense.

Yeah.

Yeah. Uh, but let me see. Uh, hand... uh

But I have a question, actually. Which tools that we expect to use to measure signals or or bring data from online?

Uh, for now, the main one you have is Movesense. Movesense. Uh...

I mean, which tools like uh vibration tools?

This... we have some tools available here, but this was the... something that I mentioned also on the others... uh on the previous time. So, Mikke, from the ISO standard uh and from one more publication, I think I... he showed in the first lecture, showed a very specific filter that is applied for vibrations. So, this kind of indicates like what vibrations matter in terms of occupational uh uh illnesses related to vibrations.

So, my suggestion is look at that vibration profile, look a bit at literature, and see what type of vibrations, like what frequencies you want to look at. And based on that, we will select the tools. Because otherwise, we have tools both here, in the maker space, in different places that we might be able to get access to. Uh, for sure, you can get like uh access to an angle grinder, uh Dremel, uh driver, and stuff like that. But my suggestion is to be a bit more scientific about that. Look which frequencies have a bad health effect, and let's get... work backwards from there in selecting the tools that we will use.

Yes, I understand.

Because maybe we will decide that, "Okay, this is not enough." Maybe we talk with someone that is like, for example, on a car shop and use one of the air bolt tighteners that they use on... because this is like lower vibration. Or we might need to find like um... how is it called? Impact hammer uh to test. So let's see from there.

But we have a few tools here. If you want just something to have like for getting started, you can drop by and I can give you one. Uh but I would prefer that you select the tools that you use based on the science behind it. Yes. Rather than the other way around, like, "This is available, so this is what we're going to use."

Um. Traditional accelerometer... Tu tu tu. Traditional motion... Human activity recognition... (muttering, looking something up) Uh... tu tu tu. (muttering)

I'll put one in uh the chat. Uh, just for you to take a look at. I don't think this includes a data set. Uh, give me a second, sorry.

Tu tu tu. Uh, so take a look at that. From general research into activity recognition or smartwatches, uh, there is a lot of data sets generated from that direction. So, look a bit into that. If you look into it and I cannot find something, then get back to me and I can take a look as well.

Uh, tu tu. Okay.

Yes, but I think it's like uh we have for now. I don't know if G.O. or Ibrahim have something else.

I don't.

That's all.

Okay. That's good. Also, one thing to take a note on, especially if you're interested in like the... GitHub and software and having things on online that you might be referenced at. It might be a side idea on the project, especially if you collect data, to make that data available freely afterwards. Because it might be used, it might be referenced. Uh, so to create... like, if you see there is a gap in what is available online in terms of data sets, to say, "Okay, here is also a data set," and you mark it and you see how such a data set should look like.

Okay.

Yeah? Yeah, yeah, that's fine.

But good. Uh, you can pass by to collect a tool sometimes. I need to talk with Mikke about what his plan is with the laptop because he has one, I have the other one, but the one that I'm using...

I have already taken one from him.

Ah, he gave you one yesterday?

Yeah. Yeah. And I think it's it's enough for us because you have one and now I have one, so two is enough.

Good, good, good. Perfect. Do you need an iPhone as well?

No, we have...

Oh no, yeah, we need the iPhone.

Okay. Perfect. Okay. Then the only thing is when you pass by and if you want one tool to use while you're around, that's all.

Yeah. Do we need to share everything that we have done and we will do in the future, so with you or like our Git up or how?

Uh, you can use your GitHub on your own. I don't need to review it while you're working on it. One thing that I would like to look at uh in the next session that we do or before that, is the things that we said about defining kind of the project. So, the things about requirements, assumptions, main trade-offs, like why are you going with one direction and not the other. So, basic things that will pretty much uh... make your life easier down the line. Because at the end, during a presentation, uh a final presentation, people will ask you, "Okay, yes, but why didn't you do it that way?" And these things need to be answered kind of now in your requirements, assumptions, and trade-offs decisions. Yes. Have you seen trade-off tables before? Have you worked with that at all?

Uh, no.

No.

Let me... give me a second. Uh, trade-off... table. (typing) Table... This is, for example. So... tu tu tu.

So, for example, this one is like making a decision between different motor options for whatever project. Uh, and then you might have a series of criteria that are your own. They could be more engineering and implementation criteria. They could be simultaneously both engineering and like knowledge criteria, like what languages do I know how to code, uh, or like what skills do I have, or how expensive it is. And then you combine all these on the uh columns.

So here you see they have efficiency, power density, production cost, blah blah blah. And then you go for every dec- for every option that you have and you start grading. And you have the general like green, blue, yellow in this case. You don't need to have this, but you could have numbered systems or whatever. So, you have some stuff that might make it a no-go. So, for them, for example, complexity, very complex for a superconductivity motor makes it kind of not possible to go with this solution at all.

Uh, but you might have things that are like better or worse. I've used it with like a zero, being it's a no-go, to 10, which is like the perfect solution for the different criteria. I've used it with colors. But do something similar to validate and explain your decision-making process. And this type of charts, keep in mind that they should be in your report at the end as well.

But good. So this is kind of a decision-making uh tool. Whenever you need to figure out how to do something, for example, what way am I going to use to identify vibration? And it could be FFT, it could be like an AI classifier. You will need to figure out, first of all, what are the options for making that decision, like what alternatives do I have, and then grading them based on the criteria that you deem are appropriate.

But let's see at least the first version of this thing, at least for some decisions, in the next meeting. Good?

Yes.

Perfect.

Do we have... Do we have other questions, guys?

No. No, for now, no.

Perfect. Thank you for today. Have a nice rest of your Friday and a nice weekend.

Thank you very much.

Thank you.

Bye.

Bye-bye.

Bye."
