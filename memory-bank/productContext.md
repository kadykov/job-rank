# Product Context: Job Description Ranker

**1. Problem Domain:**
Job seekers often face a large volume of job descriptions (JDs) when searching for opportunities. Evaluating the relevance of each JD against their own CV is a manual, time-consuming, and often imprecise process. Recruiters also face challenges in quickly identifying the most suitable candidates from a pool based on CVs against a specific JD. While this tool initially focuses on the job seeker's perspective (ranking JDs for *their* CV), the underlying technology could be adapted.

**2. User Needs:**
- **Job Seekers:** Need a faster, more accurate way to identify JDs that genuinely align with their skills and experience, allowing them to focus their application efforts. They need to understand *why* a job is considered relevant.
- **(Future) Recruiters:** Need efficient ways to screen CVs against a specific job opening.

**3. Proposed Solution & Value:**
The system aims to provide a relevance score by comparing the user's CV not directly to the JD, but to an "ideal" CV generated *from* the JD using an LLM. This approach leverages the LLM's ability to understand the nuances of requirements expressed in the JD and synthesize a profile that perfectly matches it. Comparing the user's actual CV to this "ideal" profile via embeddings and cosine similarity should provide a more meaningful measure of fit than simple keyword matching.

**Value Proposition:**
- **Efficiency:** Saves users significant time by automatically ranking JDs.
- **Accuracy:** Aims for a more nuanced understanding of relevance than keyword matching.
- **Focus:** Helps users prioritize applications for the most promising opportunities.

**4. User Experience Goals (Initial MVP):**
- Simple command-line interface.
- Clear input requirements (CV path, JD path/directory).
- Understandable output (ranked list of JD identifiers with scores).
- Minimal configuration required (primarily API key setup).
