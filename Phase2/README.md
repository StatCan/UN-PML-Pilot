# Private Machine Learning Track - Phase 2

## PET-based remote analytics service

### Scenario
An NSI wants to offer a PET-based remote analytics service, like e.g. a predictive model trained on its own internal data. The service could be used by another subject (public, private or another NSI) that needs to use an analytics service accessing NSI’s data with privacy requirements and possibly also making use of confidential data of the subject itself.

### Use Case
A university must plan study courses for the next few years and must estimate the residence time (numbers of years in which students will attend university courses) of its new students, in order to assess the number of teachers and classrooms required. 

To estimate the residence time of its new students, the university needs to use sensitive data on their socio-demographic characteristics; these data are held by an NSI. 

The NSI is available to allow the university to access its database on socio-demographic characteristics, at micro-data level, in a privacy preserving way only.

The university also has a privacy requirement in relation to its students, so there is no chance that the NSI will perform a computation on its own and just send the results to the university. 

The university will train a private ML model that estimates the average residence time of students at the university, based on their parents' educational qualifications (diploma, degree, etc…), the area of provenience (e.g. center or suburbs of a big town), and grade category inference. 

Each year, the university will be able to use the pre-trained private machine learning model to estimate the average residence time of its new students and use this statistic to plan the number of teachers and classrooms needed to provide study courses. 

### Scope
- The university has no access to the sensitive socio-demographic data. Only by the use of PETs the university can have access to the analytical output.
- Making data available for other organizations (private or public).


