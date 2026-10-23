Description
Business Context

"Visit with Us," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.
Objective

As an MLOps Engineer at "Visit with Us," your responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline will include data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment, ensuring consistent performance and scalability. By leveraging GitHub Actions for CI/CD integration, the system will enable automated updates, streamline model deployment, and improve operational efficiency. This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.
Data Dictionary

The dataset contains customer and interaction data that serve as key attributes for predicting the likelihood of purchasing the Wellness Tourism Package. The detailed attributes are:

Customer Details

CustomerID:Unique identifier for each customer.
ProdTaken:Target variable indicating whether the customer has purchased a package (0: No, 1: Yes).
Age:Age of the customer.
TypeofContact:The method by which the customer was contacted (Company Invited or Self Inquiry).
CityTier:The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
Occupation:Customer's occupation (e.g., Salaried, Freelancer).
Gender:Gender of the customer (Male, Female).
NumberOfPersonVisiting:Total number of people accompanying the customer on the trip.
PreferredPropertyStar:Preferred hotel rating by the customer.
MaritalStatus:Marital status of the customer (Single, Married, Divorced).
NumberOfTrips:Average number of trips the customer takes annually.
Passport:Whether the customer holds a valid passport (0: No, 1: Yes).
OwnCar:Whether the customer owns a car (0: No, 1: Yes).
NumberOfChildrenVisiting:Number of children below age 5 accompanying the customer.
Designation:Customer's designation in their current organization.
MonthlyIncome:Gross monthly income of the customer.
Customer Interaction Data

PitchSatisfactionScore:Score indicating the customer's satisfaction with the sales pitch.
ProductPitched:The type of product pitched to the customer.
NumberOfFollowups:Total number of follow-ups by the salesperson after the sales pitch.-
DurationOfPitch:Duration of the sales pitch delivered to the customer.
Submission Guidelines

Kindly go through the instructions below before attempting the project.

The project submission should be asingle Python notebook in HTML (.html) format.
In case Jupyter Notebook is being used, the Python notebook can be exported as an HTML file from the File menu.
In case Google Colab (preferable) is being used, the Python notebook needs to be first downloaded as a .ipynb file from Colab and can then be converted and exported as an HTML file using this link.
Any assignment foundcopied/plagiarized from any source will not be evaluated and awarded zero marks.
Please ensure timely submission asany submission post-deadline will either not be accepted for evaluation or attract penalties.
Important Points to Note

Thoroughly read the problem statement and project rubric before starting to work on the project.

Apply the concepts and techniques you have learned in the previous weeks and summarise your insights at the end.

Please run the code from start to end sequentially,add the observations and insights as applicable, and then submit the file in the mentioned format.

A template notebook has been provided for reference. Please refer to it in case you need assistance with certain sections of the project.

Before submitting the file, please make sure that all the sections mentioned in the rubric have been covered and the code outputs are clearly visible in your submission.

Add the link to the Hugging Face Spaces for the frontend of the deployed model in your notebook also add the link for the GitHub repository. Please ensure that the spaces are public.
If the code in the submission is correct but not executed, 50% of the marks will be deducted from the corresponding rubric section.

Kindly connect with the Program Manager in case of any confusion.

Power Ahead!