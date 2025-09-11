import re
from src.skills_db import skills_db

def skills_extraction(text, predicted_role):
    """Extract matching skills from resume text based on role"""
    role_skills = skills_db.get(predicted_role, []) #role keliye req skills list extract from skills_db
    found_skills = [skill for skill in role_skills if skill.lower() in text.lower()] # if skill present in resume text then add to found_skills
    return found_skills

def fit_score_computation(prob, skills_found, role):
    """Compute fit score = model probability + skill match weight"""
    prob_score = prob * 100 
    skill_weight = len(skills_found) / max(1, len(skills_db.get(role, []))) * 100 
    return round((0.7 * prob_score) + (0.3 * skill_weight), 2) # 70% weight to prob, 30% to skill match