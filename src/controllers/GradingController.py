import logging
import json
from typing import List
from routes.schemas.grading import GradingRequestBody, GradedResult, GradingAIInternalResult

logger = logging.getLogger('uvicorn.error')

class GradingController:
    def __init__(self, generation_client):
        self.generation_client = generation_client

    async def evaluate_exam(self, request_data: GradingRequestBody) -> List[GradedResult]:
        results = []
        
        for question in request_data.questions:
            # Skip grading if it's not an essay/short answer (Multiple choice should be auto-graded by the backend)
            if question.type not in ["essay", "short_answer"]:
                continue

            system_prompt = """
            You are an objective, strict university professor grading an exam. 
            You will be provided with a Question, the Expected Answer, a Grading Rubric, and the Student's Answer.

            CRITICAL RULES:
            1. STRICTNESS: Base your score ONLY on the Expected Answer and Grading Rubric. 
            2. NO SYMPATHY POINTS: Do not award points for well-written essays if they lack correct technical facts. If irrelevant, score 0.
            3. BLIND GRADING: You do not know the student. Grade solely on the text provided.
            4. BOUNDARIES: The maximum possible score is strictly defined. Never exceed it.

            You MUST output a valid JSON object matching this schema perfectly:
            {
                "reasoning_process": "String. Step-by-step evaluation of missing concepts, misconceptions, and rubric alignment.",
                "points_earned": "Float. The final calculated score.",
                "feedback": "String. Constructive feedback explaining the score to the student."
            }
            """

            user_prompt = f"""
            Question: {question.question_text}
            Expected Answer: {question.expected_answer}
            Grading Rubric: {json.dumps(question.grading_rubric) if question.grading_rubric else "No specific rubric. Grade based strictly on the Expected Answer."}
            Max Score: {question.max_score}
            
            Student's Answer: {question.student_answer}
            """

            try:
                # Call the LLM with JSON mode enforced
                llm_response = await self.generation_client.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=GradingAIInternalResult.schema() # Assuming your client supports passing schemas
                )
                
                # Parse the LLM's internal output
                parsed_response = GradingAIInternalResult.parse_raw(llm_response)
                
                # Safety check: Ensure the AI didn't hallucinate a score higher than the max
                final_score = min(float(parsed_response.points_earned), float(question.max_score))
                final_score = max(final_score, 0.0) # Ensure no negative scores

                results.append(GradedResult(
                    exam_question_id=question.exam_question_id,
                    points_earned=final_score,
                    feedback=parsed_response.feedback
                ))

                logger.info(f"Graded Question {question.exam_question_id} | Score: {final_score}/{question.max_score} | Reasoning: {parsed_response.reasoning_process}")

            except Exception as e:
                logger.error(f"Failed to grade question {question.exam_question_id}: {str(e)}")
                # Provide a fallback result so the whole exam doesn't crash if one question fails
                results.append(GradedResult(
                    exam_question_id=question.exam_question_id,
                    points_earned=0.0,
                    feedback="System encountered an error grading this response. Awaiting manual review."
                ))

        return results