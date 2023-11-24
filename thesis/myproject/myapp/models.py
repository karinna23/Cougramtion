from django.db import models

class PredResults(models.Model):
        first_name = models.CharField(max_length=100, null=True)
        last_name = models.CharField(max_length=100, null=True)
        sex = models.CharField(max_length=100, null=True)
        cet = models.FloatField()
        gpa = models.FloatField()

        STRAND_CHOICES = [
            ("0", "Accountancy, Business, and Management (ABM)"),
            ("1", "General Academic Strand (GAS)"),
            ("2", "Humanities and Social Sciences (HUMSS)"),
            ("3", "Sports Track (SP)"),
            ("4", "Science, Technology, Engineering, and Mathematics (STEM)"),
            ("5", "Technology Vocational and Livelihood (TVL)"),
            ("6", "Arts & Design (AD)"),
            

        ]
        
        strand = models.CharField(
            max_length=1,
            choices=STRAND_CHOICES,
        )

        recommended_course = models.TextField()
        

        def __str__(self):
            return self.recommended_course
    

class RecommendedCourse(models.Model):
    prediction_id = models.ForeignKey(PredResults, on_delete=models.CASCADE)
    course = models.CharField(max_length=255)
    percentage = models.CharField(max_length=155)
    description = models.TextField()
    analysis = models.TextField(null=True)