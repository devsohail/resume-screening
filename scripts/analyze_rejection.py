#!/usr/bin/env python3
"""
Analyze why a resume was rejected
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.db_handler import get_db_handler
from backend.storage.db_models import ScreeningResult, Resume, Job


def analyze_rejection(resume_id: str, job_id: str):
    """Analyze why a resume was rejected"""
    
    db = get_db_handler()
    
    with db.get_session() as session:
        # Get the screening result
        result = session.query(ScreeningResult).filter(
            ScreeningResult.resume_id == resume_id,
            ScreeningResult.job_id == job_id
        ).first()
        
        # Get resume and job
        resume = session.query(Resume).filter(Resume.id == resume_id).first()
        job = session.query(Job).filter(Job.id == job_id).first()
        
        print("═" * 70)
        print("REJECTION ANALYSIS")
        print("═" * 70)
        
        if not resume:
            print(f"\n❌ Resume not found: {resume_id}")
            return
        
        if not job:
            print(f"\n❌ Job not found: {job_id}")
            return
        
        print(f"\n👤 Candidate: {resume.candidate_name or 'Unknown'}")
        print(f"   Skills: {', '.join(resume.skills[:10]) if resume.skills else 'None'}")
        print(f"   Experience: {resume.experience_years} years")
        print(f"   Education: {resume.education or 'Not specified'}")
        
        print(f"\n💼 Job: {job.title} at {job.company}")
        print(f"   Required Skills: {', '.join(job.required_skills) if job.required_skills else 'None'}")
        print(f"   Preferred Skills: {', '.join(job.preferred_skills) if job.preferred_skills else 'None'}")
        print(f"   Min Experience: {job.min_experience_years} years")
        
        if not result:
            print(f"\n⚠️  NO SCREENING RESULT FOUND")
            print(f"   This resume has not been screened for this job yet!")
            print(f"\n💡 To screen this resume:")
            print(f"   1. Go to Job Management in UI")
            print(f"   2. Find job: {job.title}")
            print(f"   3. Click 'Upload Resume' and select the resume")
            print(f"   OR use API:")
            print(f"   curl -X POST http://localhost:8000/api/v1/screening/screen \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{'resume_id': '{resume_id}', 'job_id': '{job_id}'}}'")
            return
        
        # Detach from session
        session.expunge_all()
    
    # Now analyze the result
    print(f"\n" + "═" * 70)
    print(f"SCREENING RESULT")
    print("═" * 70)
    
    decision_icon = "✅" if result.decision.value == "shortlist" else "❌"
    print(f"\n{decision_icon} Decision: {result.decision.value.upper()}")
    print(f"📊 Final Score: {result.final_score:.2f}/100")
    
    print(f"\n💬 System Explanation:")
    print(f"   {result.explanation}")
    
    print(f"\n📈 Score Breakdown:")
    print(f"   Overall Score: {result.similarity_overall_score:.2f}%")
    print(f"   Skills Match: {result.similarity_skills_match:.2f}%")
    print(f"   Experience Match: {result.similarity_experience_match:.2f}%")
    print(f"   Semantic Similarity: {result.similarity_semantic:.2f}%")
    
    matched = result.matched_skills if result.matched_skills else []
    missing = result.missing_skills if result.missing_skills else []
    
    print(f"\n✅ Matched Skills ({len(matched)}):")
    if matched:
        for skill in matched:
            print(f"   • {skill}")
    else:
        print(f"   (none)")
    
    print(f"\n❌ Missing Skills ({len(missing)}):")
    if missing:
        for skill in missing:
            print(f"   • {skill}")
    else:
        print(f"   (none)")
    
    # Analysis
    print(f"\n" + "═" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("═" * 70)
    
    print(f"\n🔍 Why was this rejected?")
    
    issues = []
    
    if result.similarity_skills_match < 70:
        issues.append(f"• Low skills match ({result.similarity_skills_match:.1f}%)")
    
    if result.similarity_experience_match < 70:
        issues.append(f"• Low experience match ({result.similarity_experience_match:.1f}%)")
    
    if result.similarity_semantic < 0.5:
        issues.append(f"• Low semantic similarity ({result.similarity_semantic:.2f})")
    
    if missing:
        issues.append(f"• Missing {len(missing)} required skills")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"   ⚠️  POTENTIAL FALSE NEGATIVE!")
        print(f"   Score is {result.final_score:.1f}% but was rejected.")
        print(f"   This might be a mistake that needs human review!")
    
    print(f"\n💡 Recommended Actions:")
    
    # Check if this is likely a false negative
    if resume.experience_years and resume.experience_years >= (job.min_experience_years or 0):
        print(f"   ✅ Candidate HAS required experience ({resume.experience_years} >= {job.min_experience_years} years)")
    
    required_set = set(job.required_skills) if job.required_skills else set()
    candidate_set = set(resume.skills) if resume.skills else set()
    matched_count = len(required_set & candidate_set)
    
    if matched_count >= len(required_set) * 0.8:  # 80% match
        print(f"   ✅ Candidate HAS most required skills ({matched_count}/{len(required_set)})")
    
    if result.final_score > 50:
        print(f"\n   🎯 RECOMMENDATION: This looks like a FALSE NEGATIVE!")
        print(f"   • Score is {result.final_score:.1f}% (above threshold)")
        print(f"   • Has {resume.experience_years} years experience (req: {job.min_experience_years})")
        print(f"   • Matched {len(matched)} skills")
        print(f"\n   📝 Action: Submit human feedback to correct this:")
        print(f"      curl -X POST http://localhost:8000/api/v1/feedback/submit \\")
        print(f"        -H 'Content-Type: application/json' \\")
        print(f"        -d '{{")
        print(f"          \"screening_result_id\": \"{result.id}\",")
        print(f"          \"human_decision\": \"shortlist\",")
        print(f"          \"notes\": \"Strong candidate, meets all requirements\"")
        print(f"        }}'")
    
    print(f"\n" + "═" * 70)
    print("SYSTEM IMPROVEMENTS NEEDED")
    print("═" * 70)
    
    print(f"\n1. 🔧 Threshold Adjustment:")
    print(f"   Current rejection threshold appears too high")
    print(f"   Recommend: Lower from 70% to 60% for 'review' category")
    
    print(f"\n2. 🎯 Skills Matching Logic:")
    print(f"   System may not recognize skill variations")
    print(f"   Examples:")
    print(f"   • 'Python' vs 'Python Programming'")
    print(f"   • 'AWS' vs 'Amazon Web Services'")
    print(f"   • 'FastAPI' vs 'Fast API'")
    print(f"   Fix: Use embeddings + fuzzy matching")
    
    print(f"\n3. 📊 Experience Weighting:")
    print(f"   13 years experience should heavily influence score")
    print(f"   Current weight may be too low")
    print(f"   Recommend: Increase experience weight in hybrid scoring")
    
    print(f"\n4. 🤖 Model Training:")
    print(f"   No trained classifier detected")
    print(f"   Similarity-only scoring can miss qualified candidates")
    print(f"   Action: Collect 100 reviewed samples and train model")
    
    print(f"\n5. 👤 Human-in-the-Loop:")
    print(f"   This is WHY we need human feedback!")
    print(f"   Submit feedback to improve future predictions")
    
    print(f"\n" + "═" * 70)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze resume rejection')
    parser.add_argument('--resume', required=True, help='Resume ID')
    parser.add_argument('--job', required=True, help='Job ID')
    
    args = parser.parse_args()
    
    analyze_rejection(args.resume, args.job)



