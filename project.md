Project Instructions
(61.502 Deep Learning for Enterprise, Y2026)
1. Introduction
In this project, you may work on a topic of your own choice (assuming it fulfils a minimal list
of requirements detailed below). You will have to design your custom model and possibly
create your own mock dataset (or better, look for one online). I advise to have a look at the
typical dataset repository websites such as Kaggle (Google dataset search engine), as to
not waste too much time on creating or processing a dataset (this is not the purpose of this
project!). Please assume that this project is to be submitted to a serious client (played by us
professors) and take this project as an opportunity to work on your project delivery skills.
More specifically, pay attention to the way you code and document your code. Your
submission will have to be uploaded for grading. Please only submit your code and notebooks
on eDimension, not the dataset! (Otherwise, the submission will be too large). I also advise
you to upload your final project on Github, to enrich your personal portfolio of projects. Shall
the dataset be required to run your project, let me know where to download it (Kaggle or even
better, upload it to Google Drive/Dropbox). Finally, I expect you to present the project in Week
13, time to be confirmed closer to the date.
2. TL;DR
• Groups of 3 students, not allowed to work alone.
• GPU access on the SUTD AI Mega Cluster. Refer to instructions manual for details
• Username: studentID
• Password: AIcluster#2025
• Students will need to change their password after first login
• Free choice on project proposal, custom project proposal deadline and groups formation
before February 27th, 11.59pm.
• Use link for group formation:
https://docs.google.com/spreadsheets/d/1Q_ZWzAFzG_2mFpMIvcIkQySlIo6aqh4lzcY
TY3-DA8E/edit?gid=1394846873#gid=1394846873
• Project Submission deadline: Apr 17th, 11.59pm.
3. Project Grading Rubrics
Technical Implementation (50%):
• Concept and Relevance: [5%]
• Thorough understanding of relevant concepts and techniques demonstrated in
the implementation.
• Coding: [25%]
• Code must be well-structured, efficient, and thoroughly documented.
• Reproducibility of code
• Performance & Evaluation: [20%]
• Hyper-parameter tuning and comparison of results to some baseline (reference
to understand the quality of results)
• The model’s effectiveness should be evaluated using appropriate metrics:
 Classification problems → Accuracy, Precision-Recall, F1-score
 Regression problems → RMSE, MAE
 Generative models → FID, Inception Score
Creativity and Innovation (5% bonus😊😊):
• Demonstrates creative solutions or innovative approaches beyond state-of-the-art.
• Unique features or functionalities add significant value to the project.
Presentation and Communication (20%):
• Clear and well-organized presentation.
• Effective communication of project objectives, methodology, and results.
• Demonstration of the impact or potential value of the project (achievement of stated
objectives).
Project Report (30%):
• Project Report:
o Introduction: Clearly defined problem statement with well-defined objectives,
scope, and constraints.
o Method: Clear presentation of the architecture and model used for the
implementation. You can add a section containing all tried but failed methods
to demonstrate the difficulty level of the chosen problem and efforts made.
o Experiments: Clarity of specifications used and comparison of results to
demonstrate the efficacy.
o Project Link to GitHub :
 Upload the code to eDimension and github (public accessible)
 Upload dataset and the weights files to Google Drive/Dropbox if heavy
o Conclusion
4. Objectives and guidelines for project proposal
Every group will need to submit the project proposal via email to me (pritee_agrawal@sutd.edu.sg)
by February 27th, 11.59pm for approval.
The proposal should be a small PDF file, containing a brief description of the following elements:
• Topic, problem to be investigated,
• Expected inputs and outputs, dataset to be used,
• Team members,
• What you are going to deliver,
Project requirements: There is no theme constraint for this project. Please make sure that you
are using deep learning model for your project.
You are rather open to choose any crazy topics!
1. List of expectations for projects
As mentioned before, you are expected to deliver a project, as if us professors were
professional clients. As such, good practices will be critical, for instance:
• The repository is well-structured and well-documented.
• Usage and installation instructions are clear.
• Code is well-organized and documented.
• Code is reproducible, extensible, and modular.
• Create an end-to-end ML pipleline and include a pipeline diagram in the report.
• Train your model using a training set but evaluate your performance after training on
a test set. Even better, use a train-test-validation split.
• Compare the results for few different models and choose the best performing model
• Visualization of your model performance (e.g. accuracy and loss curves, performance
of your system on some validation set data, etc.) are also expected. For each figure
used in your report, there should be a clear description on how to recreate said
figure. Being unable to do so, means that there is no way for us to confirm the results
you are presenting!
• Please show examples of your model malfunctioning if any and discuss what might
be the reason for such problems.
• Your report should contain everything we need to know to run your code (including
the package dependencies).
• Put in your submission the group members and their contribution to the project.
5. Project Delivery
Delivery details
Groups of 3 people, free choice on project proposal but will not allow people to work alone.
Custom project proposal deadline and groups formation before February 27th, 11.59pm. Use
link for group formation:
https://docs.google.com/spreadsheets/d/1Q_ZWzAFzG_2mFpMIvcIkQySlIo6aqh4lzcYTY3-
DA8E/edit?gid=1394846873#gid=1394846873
Submission deadline: April 17th, 11.59pm.
Recapitulative report
Your recapitulative report shall be submitted in a PDF format, along with your code.
Code submission: Your code may consists of .py files, .ipynb jupyter notebooks, or Google
Colab notebooks.
• Properly documented notebooks/code files would be much appreciated, and you should
practice it anyway (they are good practice!). I believe a presentable format is:
o A Jupyter Notebook, combining MarkDown cells and code cells,
o Along with .py files containing the largest parts of your code (you just have to
import them later in your Jupyter Notebook, to minimize the amount of code in
your Notebook!) But ultimately, the choice is yours!
• Environment & reproducibility
o Provide requirements.txt/environment.yml (or pyproject.toml) and optional
Dockerfile.
o Document exact commands to reproduce the pipeline and figures; include
seeds and hardware notes.
• Serving/Deployment
o Lightweight demo (FastAPI/Flask or batch script) showing how the artifact
would be consumed. Mention monitoring needs (latency, drift) if deployed.
Report Submission: The final project report should be approximately 10-15 pages in length
(excluding appendix and references) and cover the following topics:
1. Executive Summary: Succinctly describe the project, results, and recommendations.
The executive summary should not exceed 1 page in length.
2. Background and Introduction: This section motivates the problem, explains why it's
important, why should we care, and the potential impact if it's solved.
3. Related work (if new model is proposed): What's been done before in this area and
why your work is different or better.
4. Problem formulation and Overview of your solution
5. Data Description, including briefly highlighting any data exploration that informed
important formulation/modeling choices.
6. Details of your solution: methods, tools, analysis you did, model types and
hyperparameters used, features. This section of the report should also include a link
to well-documented code in your group’s course github repository. Your report should
explicitly mention what needs to be done to run your code (imports needed, where to
find dataset, commands to run if any, etc.). 
7. Evaluation: results, plots (for example precision recall k curves, other types of results),
important features, and bias audit of the models you built.
8. Discussion of the results: what did you learn from looking at the results about the data,
problem, and solution.
9. Any recommendations based on your analysis/models
10. Limitations, caveats, future work to improve on what you've done.
11. Optionally, you may also wish to include a proposal for future avenues of research
beyond the scope of this work, for instance on novel machine learning methods to
improve on the current work or other related research opportunities.
Project delivery
We strongly advise to upload your submission (code/notebooks + PDFs, but no dataset due to
space restrictions) on a Github repository. You can then submit the link to your public Github
repository, during your submission on edimension.
• Your Github repository for this project should contain your PDF report, your
DOCUMENTED code/notebook files. It should also contain directions showing the
required libraries and steps needed to re-train the model from scratch. And more
importantly, it should also contain clear directions on how to recreate the exact trained
model and its performance results you are presenting in the PDF. This is essential, for
reproducibility reasons.
Project presentation
Finally, we expect you to present the project during one of the sessions of Week 13. A small
demo, along with some slides or a small video would be appreciated. Time and details to be
confirmed closer to the date.