# Football Shot Analyst Framework

## Model Card

### **Model Overview**
- **Name:** Football Shot Analyst (xG Prediction + Language Model)
- **Version:** 1.0
- **Developers:** The football Shot Analyst is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational), customized by Pegah Rahimian and David Sumpter as a part of Uppsala University research, and
is intended as an illustration of the methods. This work is a derivative of the full [Twelve GPT product](https://twelve.football).
- **Date of Creation:** [Autumn 2024]
- **Purpose:** 
  - Here we detail the specific application to analyze shots. It is thus intended as an example to help others building wordalisations. The wordalisations describe shots attempted by players with different outcomes (goal or not) in EURO 2024.
  - Predict the quality of football shots using an xG model.
  - Translate the numeric xG predictions and associated features into engaging and informative textual descriptions using a language model.

---

### **Intended Use**
- **Primary Use Cases:**
  - Analyzing football match data to assess shot quality and contributing factors.
  - Generating automated yet human-readable summaries of football shots for analysts, commentators, or fans.
- **Target Users:**
  - Sports analysts and coaches.
  - Football commentators seeking quick insights during live games.
  - Enthusiasts exploring data-driven insights into football.

---

### **Model Architecture**
- **xG Prediction Model:**
  - **Framework:** [e.g., Logistic Regression, Random Forest, Neural Network, etc.]
  - **Features:** 
    - Vertical distance to goal center.
    - Euclidean distance to goal.
    - Angle of the shot.
    - Body part used to take the shot (e.g., header, left foot, etc.)
    - Pressure metrics (e.g., proximity of nearest opponent, number of ooponents blocking the path, etc.).
    - Contextual information (e.g., set-piece situations).
  - **Output:** xG value (range: 0 to 1), indicating the probability of scoring.
  
  
- **Language Model for Descriptions:**
  - **Framework:** The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. 
  - **Input:** Synthetic textual prompts derived from xG model predictions and contributing factors.
  - **Output:** Structured, engaging narratives about the shots.

---

### **Training Data**
- **xG Prediction Model:**
  - **Source:** The datasets used in this work was obtained from the Hudle StatsBomb events and StatsBomb360 datasets for [UEFA EURO 2024](https://statsbomb.com/news/statsbomb-release-free-euro-2024-data/), fetched using the [statsbombpy API](https://github.com/statsbomb/statsbombpy).
  - **Dataset:** The StatsBomb events dataset contains 110 columns describing various aspects of each event, while the StatsBomb360 dataset includes 7 columns detailing the position of players visible in the frame of the action. These datasets were merged to provide a comprehensive view of the events for all matches played by the 24 teams participating in UEFA EURO 2024.
  - **Preprocessing:** Normalized pitch coordinates, synchronising events with freeze frames (location of all visible players in the frame), engineered features for distance/angle metrics, and exclusion of ambiguous events.
  
- **Prompts for the Language Model:**
  - To generate engaging and contextually relevant descriptions of xG (Expected Goals) predictions, we rely on structured prompts and percentile-based categorization of xG values. These structured examples guide the model in transforming numerical predictions into natural language that can captivate the audience.
  - The prompt to *tell it who it is* identifies a human role for the wordalisation as a "data analysis bot". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted in footballing terms](https://github.com/soccermatics/twelve-gpt-educational/blob/main/data/describe/action/shots.xlsx). These descriptions outline the meaning of the footballing terms.
  - In the text which is generated at the stage of *tell it what data to use* we use the following function to translate numbers and thresholds to evaluation words:
    ```python
    def describe_xg(xG):

        if xG < 0.028723: # 25% percentile
            description = "This was a slim chance of scoring."
        elif xG < 0.056474: # 50% percentile
            description = "This was a low chance of scoring."
        elif xG < 0.096197: # 75% percentile
            description = "This was a decent chance."
        elif xG < 0.3: # very high
            description = "This was a high-quality chance, with a good probability of scoring."
        else:
            description = "This was an excellent chance."
        
        return description
    ```
  - In the *tell it how to answer* step, we ask the language model the following:
    ```python
    def get_prompt_messages(self):
        prompt = (
            "You are a football commentator. You should write in an exciting and engaging way about a shot"
            f"You should giva a four sentence summary of the shot taken by the player. "
            "The first sentence should say whether it was a good chance or not, state the expected goals value and also state if it was a goal. "
            "The second and third sentences should describe the most important factors that contributed to the quality of the chance. "
            "If it was a good chance these two sentences chould explain what contributing factors made the shot dangerous. "
            "If it wasn't particularly good chance then these two sentences chould explain why it wasn't a good chance. "
            "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when shooting."
            )
        return [{"role": "user", "content": prompt}]
    ```
  - Example Outputs:
    > A goal! Wirtz finds the back of the net with a shot that, while only a 15% chance according to xG (0.15), proved deadly accurate. The central location of the shot, right down the middle, significantly increased its probability, while the lack of immediate pressure allowed him the time and space to pick his spot. Despite multiple defenders in the way, Wirtz showed composure and skill beyond his years! A truly magnificent finish!

---

### **Evaluation**
- **xG Prediction Model:**
  - **Accuracy:** [ROC AUC = 0.85].
  - **Performance across subgroups:** Tested for different leagues and playing styles to ensure robustness.

- **Language Model:**
  - TODO(Evaluation via user surveys assessing clarity, engagement, and correctness.)
  - TODO(Metrics: Average user rating (4.7/5 for clarity), BLEU score for coherence.)

---

### **Limitations**
- **xG Prediction Model:**
  - Dataset bias: Performance may degrade for leagues or matches with sparse data.
  - Feature sensitivity: Model relies heavily on accurate event tagging and player tracking.
  
- **Language Model:**
  - Generalization limits: May produce repetitive or overly generic narratives for uncommon scenarios.
  - Dependency on prompt quality: Requires carefully crafted input for optimal performance.

---

### **Ethical Considerations**
- **Fairness:**
  - Potential biases in data (e.g., underrepresentation of certain leagues or teams) could affect model outputs.
  
- **Transparency:**
  - Clear communication of xG predictions and their uncertainties is essential to avoid misinterpretation.

- **Privacy:**
  - All data used in the framework complies with relevant privacy laws and anonymization standards.

---

### **Caveats and Recommendations**
- Ensure the input data is preprocessed and normalized according to the model's requirements.
- Use the generated text descriptions as supplementary insights, not as definitive analyses.

---

### **Example Visualization and Output**

**Figure 1:** Visualization of shot situations and generated text output.

- **Description:** 
  - The figure includes a graphical representation of shot scenarios overlaid on a football pitch, highlighting critical features such as shot location, nearest opponents, and event context. 
  - Alongside, an example of synthesized text demonstrates how the xG predictions are verbalized for end users.

---

## License
TODO(This framework is licensed under [Your License]. Please refer to the `LICENSE` file for more details.)
