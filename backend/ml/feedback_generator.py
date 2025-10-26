"""
Generate improvement feedback for candidates
Tells candidates what they need to improve to match the job
"""

from typing import List, Dict, Optional


def generate_improvement_feedback(
    decision: str,
    final_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    required_skills: List[str],
    preferred_skills: List[str],
    candidate_experience: Optional[float],
    required_experience: Optional[int],
    similarity_scores: Dict[str, float]
) -> Dict[str, any]:
    """
    Generate detailed feedback for candidates on how to improve
    
    Returns:
        {
            "summary": "Overall feedback summary",
            "strengths": ["list of strengths"],
            "improvements": ["list of areas to improve"],
            "recommendations": ["specific action items"]
        }
    """
    
    strengths = []
    improvements = []
    recommendations = []
    
    # Analyze skills
    total_required = len(required_skills) if required_skills else 0
    matched_count = len(matched_skills) if matched_skills else 0
    missing_count = len(missing_skills) if missing_skills else 0
    
    if matched_count > 0:
        strengths.append(f"Strong match on {matched_count} key skills: {', '.join(matched_skills[:5])}")
    
    if missing_count > 0:
        improvements.append(f"Missing {missing_count} required skills")
        recommendations.append(
            f"Gain experience in: {', '.join(missing_skills)}. "
            f"Consider online courses, projects, or certifications."
        )
    
    # Analyze experience
    if candidate_experience and required_experience:
        if candidate_experience >= required_experience:
            strengths.append(f"Excellent experience level ({candidate_experience} years)")
        else:
            gap = required_experience - candidate_experience
            improvements.append(f"Need {gap} more years of relevant experience")
            recommendations.append(
                f"Highlight relevant projects and accomplishments that demonstrate "
                f"experience beyond your years. Consider contract or freelance work."
            )
    
    # Analyze similarity scores
    skills_match = similarity_scores.get('skills_match', 0)
    experience_match = similarity_scores.get('experience_match', 0)
    semantic_similarity = similarity_scores.get('semantic_similarity', 0)
    
    if skills_match < 60:
        improvements.append("Low technical skills alignment with job requirements")
        recommendations.append(
            "Review the job description carefully and update your resume to highlight "
            "relevant skills. Use keywords from the job posting."
        )
    
    if semantic_similarity < 0.5:
        improvements.append("Resume content doesn't closely match job description")
        recommendations.append(
            "Tailor your resume to this specific role. Use similar language and terms "
            "as the job description. Highlight relevant accomplishments."
        )
    
    # Decision-specific feedback
    if decision == 'reject':
        if final_score < 40:
            summary = (
                "Your profile doesn't align well with this role's requirements. "
                "Significant skill development needed."
            )
        elif final_score < 60:
            summary = (
                "Your profile shows some potential but needs improvement in key areas "
                "to be competitive for this role."
            )
        else:
            summary = (
                "You're close! A few improvements could make you a strong candidate "
                "for similar roles."
            )
    elif decision == 'review':
        summary = (
            "Your profile is interesting but needs some clarification or improvement "
            "in certain areas before making a final decision."
        )
    else:  # shortlist
        summary = "Excellent match! Your profile aligns well with the job requirements."
    
    # Preferred skills analysis
    if preferred_skills:
        preferred_set = set([s.lower() for s in preferred_skills])
        matched_set = set([s.lower() for s in matched_skills]) if matched_skills else set()
        preferred_matched = preferred_set & matched_set
        preferred_missing = preferred_set - matched_set
        
        if preferred_matched:
            strengths.append(f"Bonus: You have preferred skills like {', '.join(list(preferred_matched)[:3])}")
        
        if preferred_missing and decision != 'shortlist':
            recommendations.append(
                f"Stand out by developing these preferred skills: {', '.join(list(preferred_missing)[:3])}"
            )
    
    # General recommendations
    if decision == 'reject':
        recommendations.append(
            "Consider roles that better match your current skill set, or invest time "
            "in developing the missing skills before reapplying."
        )
        recommendations.append(
            "Network with people in this field to learn what skills are most valued "
            "and get mentorship."
        )
    
    # If no improvements found but rejected (false negative?)
    if not improvements and decision == 'reject':
        improvements.append(
            "Your profile looks strong on paper. This might be a system error or "
            "very specific role requirements. Consider reaching out directly."
        )
    
    return {
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "recommendations": recommendations,
        "score_breakdown": {
            "overall": final_score,
            "skills_match": skills_match,
            "experience_match": experience_match,
            "semantic_similarity": semantic_similarity * 100
        }
    }


def format_feedback_for_candidate(feedback: Dict) -> str:
    """
    Format feedback into a readable text for candidates
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append("CANDIDATE FEEDBACK")
    lines.append("=" * 70)
    
    lines.append(f"\n{feedback['summary']}")
    
    lines.append(f"\n📊 YOUR SCORE: {feedback['score_breakdown']['overall']:.1f}/100")
    lines.append(f"   • Skills Match: {feedback['score_breakdown']['skills_match']:.1f}%")
    lines.append(f"   • Experience Match: {feedback['score_breakdown']['experience_match']:.1f}%")
    lines.append(f"   • Content Relevance: {feedback['score_breakdown']['semantic_similarity']:.1f}%")
    
    if feedback['strengths']:
        lines.append("\n✅ YOUR STRENGTHS:")
        for strength in feedback['strengths']:
            lines.append(f"   • {strength}")
    
    if feedback['improvements']:
        lines.append("\n⚠️  AREAS FOR IMPROVEMENT:")
        for improvement in feedback['improvements']:
            lines.append(f"   • {improvement}")
    
    if feedback['recommendations']:
        lines.append("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(feedback['recommendations'], 1):
            lines.append(f"   {i}. {rec}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)

