from langgraph.checkpoint.memory import MemorySaver
from langmem import create_manage_memory_tool, create_search_memory_tool


memory_checkpointer = MemorySaver()

manage_mem = create_manage_memory_tool(namespace=("memories",), instructions="""
Proactively call this tool when you:

1. Learn or confirm personal details about the user:
   - Name or nickname, gender, age.
   - Profession, occupation, location (city, country).
   - Special circumstances (e.g. “I’m traveling,” “I’ve just moved,” “I’m studying for exams”).

2. Detect stable preferences, habits, or styles:
   - Favorite topics, genres, products, tastes.
   - Communication style: formal vs. casual, use of emojis, humor.
   - Productivity patterns: “I’m most alert in the morning,” “Evenings are for brainstorming.”

3. Receive an explicit request to remember something:
   - “Please remember …,” “Don’t forget …,” “Remind me later about ….”
   - Reminders for meetings, events, tasks or future conversational prompts.

4. Encounter a long‑term project or ongoing context:
   - Trip planning, multi‑step assignments, home renovations.
   - Follow‑up on previously discussed items when the user returns to the conversation.

5. Uncover new, important contextual details mid‑dialogue:
   - Additional constraints, criteria, or project requirements.
   - Status updates (e.g. “My draft is now complete,” “The budget changed”).

6. Identify that an existing memory is incorrect or outdated:
   - The user corrects previously stored facts.
   - You notice contradictions between stored memory and new information.

7. Need to leverage past interactions to improve response quality:
   - Address the user by name, reference earlier jokes or examples.
   - When a key fact from prior chats adds coherence or personalization.

—
Do not call this tool for:
- Ephemeral or one‑off details unlikely to matter later (e.g. “I had an apple today”).
- Non‑personal trivia that won’t influence future behavior.
""")

search_mem = create_search_memory_tool(
    namespace=("memories",),
    instructions="""
Proactively call this tool when you need to retrieve stored user memory because it’s relevant to the current response or decision‑making, especially if the information may have fallen out of your short‑term context (which spans only the last 10 turns). For example:

1. Addressing the user by name, nickname, or title.
2. Applying known preferences (topics, styles, favorite products, habits).
3. Referring back to ongoing projects or long‑term plans (trip itinerary, coursework, renovation).
4. Reusing previous constraints or criteria (budget limits, deadlines, personal requirements).
5. Ensuring consistency when the user revisits a topic after a gap longer than 10 turns.
6. Checking for updates or corrections before acting on a stored fact.
7. Personalizing your tone or suggestions based on earlier details (time zone, communication style).
8. Verifying whether a memory exists before prompting the user again (“As I recall, you prefer X—does that still hold?”).

—  
**Do not call** this tool if:
- The needed details are still present in your immediate 10‑turn context.
- The information is ephemeral or truly one‑off and not stored.
"""
)