# CrossCheck Verification Framework
## Documentation Accuracy Validation Strategies

---

## Overview

Beyond basic comparison, we introduce **active verification** — testing whether documentation is sufficient to understand, predict, and reconstruct code behavior. Agent C acts as an examiner, using multiple verification strategies to stress-test documentation quality.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VERIFICATION HIERARCHY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Level 1: PASSIVE VERIFICATION (current)                                   │
│   └── Compare A vs B outputs                                                │
│   └── Validate against static analysis                                      │
│                                                                             │
│   Level 2: ACTIVE INTERROGATION (new)                                       │
│   └── Q&A examination                                                       │
│   └── Masked reconstruction                                                 │
│   └── Scenario walkthrough                                                  │
│                                                                             │
│   Level 3: BEHAVIORAL VERIFICATION (new)                                    │
│   └── Mutation detection                                                    │
│   └── Impact analysis                                                       │
│   └── Edge case extraction                                                  │
│                                                                             │
│   Level 4: GENERATIVE VERIFICATION (new)                                    │
│   └── Code reconstruction                                                   │
│   └── Test generation                                                       │
│   └── Inverse documentation                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Verification Strategies

### Strategy 1: Q&A Interrogation

**Concept:** C generates questions about the code, A and B must answer using ONLY their documentation. Answers are compared and validated against actual code.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Q&A INTERROGATION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                           │
│   │  Agent C    │                                                           │
│   │  (Examiner) │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          │ Generates questions from code                                    │
│          │ (C has access to source code)                                    │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ Q: "What happens if process_data() receives an empty list?"    │       │
│   │ Q: "Which functions are called when validate() fails?"         │       │
│   │ Q: "What is the return type when cache_hit is True?"           │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│          │                                                                  │
│          ├────────────────────┬────────────────────┐                        │
│          ▼                    ▼                    ▼                        │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  Team A     │      │  Team B     │      │  Code Truth │                │
│   │  (docs only)│      │  (docs only)│      │  (actual)   │                │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘                │
│          │                    │                    │                        │
│          ▼                    ▼                    ▼                        │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │ A: "Returns │      │ A: "Raises  │      │ ACTUAL:     │                │
│   │  empty dict"│      │  ValueError"│      │ Returns []  │                │
│   └─────────────┘      └─────────────┘      └─────────────┘                │
│          │                    │                    │                        │
│          └────────────────────┴────────────────────┘                        │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ VERDICT:            │                                  │
│                    │ Both teams WRONG    │                                  │
│                    │ → Documentation gap │                                  │
│                    │ → Generate JIRA     │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Question Categories:**

| Category | Example Questions | Tests |
|----------|-------------------|-------|
| **Return Values** | "What does X return when Y is None?" | Completeness of return docs |
| **Side Effects** | "Does X modify any global state?" | Hidden behavior documentation |
| **Error Handling** | "What exceptions can X raise?" | Exception documentation |
| **Edge Cases** | "What happens at boundary conditions?" | Edge case coverage |
| **Dependencies** | "What must be initialized before calling X?" | Precondition docs |
| **Call Flow** | "Trace the call path from A to B" | Call graph accuracy |

**Implementation:**

```python
class QAInterrogator:
    """Generates and evaluates Q&A challenges for documentation verification."""
    
    def __init__(self, examiner_model: str = "claude-opus-4-5"):
        self.examiner = LLMClient(examiner_model)
        self.question_types = [
            "return_value", "side_effect", "error_handling",
            "edge_case", "dependency", "call_flow"
        ]
    
    async def generate_questions(
        self, 
        component: CodeComponent, 
        num_questions: int = 5
    ) -> List[Question]:
        """Generate verification questions from source code."""
        
        prompt = f"""
        You are examining this code to generate verification questions.
        The questions will test whether documentation accurately describes behavior.
        
        SOURCE CODE:
        ```{component.language}
        {component.source_code}
        ```
        
        Generate {num_questions} questions that:
        1. Have definitive answers derivable from the code
        2. Test understanding of behavior, not just syntax
        3. Cover different aspects: return values, errors, edge cases, call flow
        4. Would expose documentation gaps if answered incorrectly
        
        For each question, provide:
        - The question
        - The correct answer (from code analysis)
        - What documentation gap a wrong answer would indicate
        
        Output as JSON array.
        """
        
        response = await self.examiner.generate(prompt)
        return [Question(**q) for q in json.loads(response)]
    
    async def evaluate_answer(
        self,
        question: Question,
        team_a_answer: str,
        team_b_answer: str,
        ground_truth: str
    ) -> QAResult:
        """Evaluate team answers against ground truth."""
        
        prompt = f"""
        QUESTION: {question.text}
        
        CORRECT ANSWER (from code): {ground_truth}
        
        TEAM A ANSWER (from their docs): {team_a_answer}
        TEAM B ANSWER (from their docs): {team_b_answer}
        
        Evaluate each answer:
        1. Is it correct? (matches ground truth semantically)
        2. Is it complete? (covers all aspects)
        3. If wrong, what documentation gap does this reveal?
        
        Output JSON with: team_a_correct, team_b_correct, 
        documentation_gap (if any), severity (low/medium/high)
        """
        
        result = await self.examiner.generate(prompt)
        return QAResult(**json.loads(result))
```

---

### Strategy 2: Masked Reconstruction

**Concept:** C masks portions of the code and asks A/B teams to reconstruct what's hidden using ONLY their documentation. Tests if docs are detailed enough to understand implementation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MASKED RECONSTRUCTION FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ORIGINAL CODE:                                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ def calculate_discount(price, customer_type, quantity):        │       │
│   │     if customer_type == "premium":                              │       │
│   │         base_discount = 0.2                                     │       │
│   │     else:                                                       │       │
│   │         base_discount = 0.1                                     │       │
│   │                                                                 │       │
│   │     if quantity > 100:                                          │       │
│   │         volume_bonus = 0.05                                     │       │
│   │     else:                                                       │       │
│   │         volume_bonus = 0                                        │       │
│   │                                                                 │       │
│   │     return price * (1 - base_discount - volume_bonus)           │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│   MASKED VERSION (sent to teams):                                           │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ def calculate_discount(price, customer_type, quantity):        │       │
│   │     if customer_type == "premium":                              │       │
│   │         base_discount = ████████                                │       │
│   │     else:                                                       │       │
│   │         base_discount = ████████                                │       │
│   │                                                                 │       │
│   │     if ████████████████:                                        │       │
│   │         volume_bonus = ████████                                 │       │
│   │     else:                                                       │       │
│   │         volume_bonus = ████████                                 │       │
│   │                                                                 │       │
│   │     return ████████████████████████████████████                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│          ┌────────────────────┴────────────────────┐                        │
│          ▼                                         ▼                        │
│   ┌─────────────────┐                       ┌─────────────────┐             │
│   │ Team A          │                       │ Team B          │             │
│   │ (uses their     │                       │ (uses their     │             │
│   │  documentation) │                       │  documentation) │             │
│   └────────┬────────┘                       └────────┬────────┘             │
│            │                                         │                      │
│            ▼                                         ▼                      │
│   ┌─────────────────┐                       ┌─────────────────┐             │
│   │ RECONSTRUCTION: │                       │ RECONSTRUCTION: │             │
│   │ base = 0.2/0.1  │                       │ base = 0.15/0.1 │             │
│   │ qty > 100       │                       │ qty > 50        │             │
│   │ bonus = 0.05    │                       │ bonus = 0.1     │             │
│   └────────┬────────┘                       └────────┬────────┘             │
│            │                                         │                      │
│            └─────────────────┬───────────────────────┘                      │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │ COMPARISON:         │                                  │
│                    │ A: 2/4 correct      │                                  │
│                    │ B: 1/4 correct      │                                  │
│                    │                     │                                  │
│                    │ Both missed:        │                                  │
│                    │ - Volume threshold  │                                  │
│                    │ - Return formula    │                                  │
│                    │                     │                                  │
│                    │ → Documentation     │                                  │
│                    │   lacks specifics   │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Masking Categories:**

| Mask Type | What's Hidden | Tests |
|-----------|--------------|-------|
| **Constants** | Magic numbers, thresholds | Specific value documentation |
| **Conditions** | If/else logic, loop bounds | Business rule documentation |
| **Expressions** | Return statements, calculations | Algorithm documentation |
| **Function Calls** | Which functions are called | Call graph accuracy |
| **Error Handling** | Try/except blocks | Exception documentation |
| **Full Blocks** | Entire if/else branches | Branch behavior docs |

**Implementation:**

```python
class MaskedReconstructor:
    """Creates masked code challenges and evaluates reconstructions."""
    
    MASK_PATTERNS = {
        "constants": r'\b\d+\.?\d*\b',  # Numbers
        "strings": r'["\'][^"\']*["\']',  # String literals
        "conditions": r'if\s+(.+?):',  # Condition expressions
        "returns": r'return\s+(.+?)(?:\n|$)',  # Return expressions
    }
    
    def create_masked_challenge(
        self,
        component: CodeComponent,
        mask_types: List[str],
        mask_ratio: float = 0.3
    ) -> MaskedChallenge:
        """Create a masked version of code for reconstruction testing."""
        
        code = component.source_code
        masks = []
        
        for mask_type in mask_types:
            pattern = self.MASK_PATTERNS.get(mask_type)
            if pattern:
                for match in re.finditer(pattern, code):
                    if random.random() < mask_ratio:
                        masks.append(Mask(
                            start=match.start(),
                            end=match.end(),
                            original=match.group(),
                            mask_type=mask_type
                        ))
        
        # Apply masks
        masked_code = self._apply_masks(code, masks)
        
        return MaskedChallenge(
            component_id=component.id,
            masked_code=masked_code,
            masks=masks,
            original_code=code
        )
    
    async def evaluate_reconstruction(
        self,
        challenge: MaskedChallenge,
        team_a_reconstruction: str,
        team_b_reconstruction: str
    ) -> ReconstructionResult:
        """Evaluate how well teams reconstructed masked code."""
        
        results = {
            "team_a": self._score_reconstruction(
                challenge.masks, 
                team_a_reconstruction, 
                challenge.original_code
            ),
            "team_b": self._score_reconstruction(
                challenge.masks,
                team_b_reconstruction,
                challenge.original_code
            )
        }
        
        # Identify documentation gaps
        gaps = []
        for mask in challenge.masks:
            a_correct = results["team_a"]["mask_scores"].get(mask.id, False)
            b_correct = results["team_b"]["mask_scores"].get(mask.id, False)
            
            if not a_correct and not b_correct:
                gaps.append(DocumentationGap(
                    mask_type=mask.mask_type,
                    original_value=mask.original,
                    severity="high",
                    recommendation=f"Add specific {mask.mask_type} documentation"
                ))
        
        return ReconstructionResult(
            scores=results,
            documentation_gaps=gaps,
            overall_quality=self._calculate_quality_score(results)
        )
```

---

### Strategy 3: Scenario Walkthrough

**Concept:** C provides input scenarios and asks teams to trace execution using their documentation. Validates that documentation captures actual runtime behavior.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCENARIO WALKTHROUGH                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SCENARIO: "User calls api.create_order(items=[], user_id=None)"           │
│                                                                             │
│   TASK: Trace the execution path and predict:                               │
│   1. Which functions will be called (in order)?                             │
│   2. What will be returned or raised?                                       │
│   3. What side effects will occur?                                          │
│                                                                             │
│   ┌─────────────────┐                       ┌─────────────────┐             │
│   │ Team A Trace:   │                       │ Team B Trace:   │             │
│   │                 │                       │                 │             │
│   │ 1. create_order │                       │ 1. create_order │             │
│   │ 2. validate_usr │                       │ 2. validate_usr │             │
│   │ 3. raises       │                       │ 3. validate_itm │             │
│   │    AuthError    │                       │ 4. raises       │             │
│   │                 │                       │    EmptyCartErr │             │
│   │ Side effects:   │                       │                 │             │
│   │ - Logs warning  │                       │ Side effects:   │             │
│   │                 │                       │ - None          │             │
│   └────────┬────────┘                       └────────┬────────┘             │
│            │                                         │                      │
│            └─────────────────┬───────────────────────┘                      │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │ ACTUAL EXECUTION:   │                                  │
│                    │                     │                                  │
│                    │ 1. create_order     │                                  │
│                    │ 2. validate_user    │                                  │
│                    │ 3. raises AuthError │                                  │
│                    │                     │                                  │
│                    │ Side effects:       │                                  │
│                    │ - Logs warning      │                                  │
│                    │ - Increments metric │ ◄── MISSED BY BOTH               │
│                    └─────────────────────┘                                  │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │ VERDICT:            │                                  │
│                    │ Team A: 90% correct │                                  │
│                    │ Team B: 60% correct │                                  │
│                    │                     │                                  │
│                    │ Gap: Side effect    │                                  │
│                    │ (metrics) not       │                                  │
│                    │ documented          │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Scenario Types:**

| Type | Example | Validates |
|------|---------|-----------|
| **Happy Path** | Valid input, normal flow | Basic call graph |
| **Error Path** | Invalid input | Exception documentation |
| **Edge Case** | Boundary values | Edge case handling |
| **Concurrent** | Race conditions | Thread safety docs |
| **State-dependent** | Different initial states | Precondition docs |

---

### Strategy 4: Mutation Detection

**Concept:** C introduces subtle bugs into code and asks teams if their documentation would help a reviewer catch the bug.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MUTATION DETECTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ORIGINAL:                              MUTATED:                           │
│   ┌─────────────────────────┐            ┌─────────────────────────┐        │
│   │ if balance >= amount:   │   ──►      │ if balance > amount:    │        │
│   │     process_withdraw()  │            │     process_withdraw()  │        │
│   └─────────────────────────┘            └─────────────────────────┘        │
│                                                     │                       │
│                                                     ▼                       │
│                              ┌───────────────────────────────────────┐      │
│                              │ QUESTION TO TEAMS:                    │      │
│                              │                                       │      │
│                              │ Based on your documentation, would    │      │
│                              │ a code reviewer notice that this      │      │
│                              │ implementation is incorrect?          │      │
│                              │                                       │      │
│                              │ What specific documentation would     │      │
│                              │ help catch this bug?                  │      │
│                              └───────────────────────────────────────┘      │
│                                          │                                  │
│                   ┌──────────────────────┴──────────────────────┐           │
│                   ▼                                             ▼           │
│   ┌─────────────────────────────┐             ┌─────────────────────────────┐
│   │ Team A Response:            │             │ Team B Response:            │
│   │                             │             │                             │
│   │ "Our docs state 'withdrawal │             │ "Our docs say 'checks if    │
│   │ succeeds when balance is    │             │ sufficient balance exists'  │
│   │ at least equal to amount'   │             │ which is vague and would    │
│   │ - reviewer would catch it"  │             │ NOT catch this bug"         │
│   │                             │             │                             │
│   │ Confidence: HIGH            │             │ Confidence: LOW             │
│   └─────────────────────────────┘             └─────────────────────────────┘
│                                                                             │
│   EVALUATION: Team A's documentation is more precise and would             │
│               enable bug detection. Team B needs to improve.                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Mutation Types:**

| Mutation | Example | Tests |
|----------|---------|-------|
| **Boundary** | `>=` → `>` | Boundary condition docs |
| **Off-by-one** | `i < n` → `i <= n` | Loop documentation |
| **Wrong variable** | `x` → `y` | Variable purpose docs |
| **Missing call** | Remove function call | Call graph completeness |
| **Wrong order** | Swap statement order | Sequence documentation |
| **Null handling** | Remove null check | Error handling docs |

---

### Strategy 5: Code Reconstruction

**Concept:** Given only the documentation, can an agent write functionally equivalent code? If not, the docs are incomplete.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CODE RECONSTRUCTION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TEAM A DOCUMENTATION:                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ calculate_shipping(weight, destination, express=False)          │       │
│   │                                                                 │       │
│   │ Calculates shipping cost based on package weight and            │       │
│   │ destination zone. Express shipping adds 50% surcharge.          │       │
│   │                                                                 │       │
│   │ Parameters:                                                     │       │
│   │   weight: Package weight in kg                                  │       │
│   │   destination: Destination zone code (1-5)                      │       │
│   │   express: Whether to use express shipping                      │       │
│   │                                                                 │       │
│   │ Returns: Shipping cost in dollars                               │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ RECONSTRUCTION      │                                  │
│                    │ AGENT               │                                  │
│                    │ (generates code     │                                  │
│                    │  from docs only)    │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ GENERATED CODE:                                                 │       │
│   │                                                                 │       │
│   │ def calculate_shipping(weight, destination, express=False):     │       │
│   │     # ??? What's the base rate per kg?                          │       │
│   │     # ??? How do zones affect pricing?                          │       │
│   │     # ??? What's the minimum charge?                            │       │
│   │     base_cost = weight * UNKNOWN_RATE * destination  # ???      │       │
│   │     if express:                                                 │       │
│   │         base_cost *= 1.5                                        │       │
│   │     return base_cost                                            │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ GAPS IDENTIFIED:    │                                  │
│                    │                     │                                  │
│                    │ • Base rate missing │                                  │
│                    │ • Zone calculation  │                                  │
│                    │   not specified     │                                  │
│                    │ • Min charge not    │                                  │
│                    │   documented        │                                  │
│                    │ • Max weight limit? │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Strategy 6: Impact Analysis Challenge

**Concept:** C asks "If we change X, what else would break?" Teams must answer from documentation. Validates dependency documentation.

```python
class ImpactAnalyzer:
    """Tests whether documentation captures dependencies accurately."""
    
    async def create_impact_challenge(
        self,
        component: CodeComponent,
        change_type: str
    ) -> ImpactChallenge:
        """Generate an impact analysis challenge."""
        
        changes = {
            "signature": f"Change {component.name} signature: add required param",
            "return_type": f"Change {component.name} return type from X to Y",
            "behavior": f"Change {component.name} to return None on error instead of raising",
            "removal": f"Delete {component.name} entirely",
            "rename": f"Rename {component.name} to {component.name}_v2"
        }
        
        return ImpactChallenge(
            component_id=component.id,
            change_description=changes[change_type],
            question="Based on your documentation, list ALL components that would need to change"
        )
    
    async def evaluate_impact_response(
        self,
        challenge: ImpactChallenge,
        team_response: List[str],
        actual_impacted: List[str]  # From static analysis
    ) -> ImpactResult:
        """Evaluate team's impact prediction against actual dependencies."""
        
        predicted = set(team_response)
        actual = set(actual_impacted)
        
        return ImpactResult(
            true_positives=predicted & actual,
            false_positives=predicted - actual,  # Overclaimed
            false_negatives=actual - predicted,   # Missed - CRITICAL
            precision=len(predicted & actual) / len(predicted) if predicted else 0,
            recall=len(predicted & actual) / len(actual) if actual else 0,
            documentation_quality="good" if len(actual - predicted) == 0 else "incomplete"
        )
```

---

### Strategy 7: Adversarial Review

**Concept:** One team reviews the other team's documentation and tries to find errors by checking against code. Incentivizes thoroughness.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADVERSARIAL REVIEW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────┐                                  │
│                    │     Agent C         │                                  │
│                    │   (Coordinator)     │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                             │
│              ┌────────────────┴────────────────┐                            │
│              ▼                                 ▼                            │
│   ┌─────────────────────┐           ┌─────────────────────┐                 │
│   │ Team A              │           │ Team B              │                 │
│   │ Reviews Team B's    │◄─────────►│ Reviews Team A's    │                 │
│   │ documentation       │           │ documentation       │                 │
│   │                     │           │                     │                 │
│   │ Has access to:      │           │ Has access to:      │                 │
│   │ - B's docs          │           │ - A's docs          │                 │
│   │ - Source code       │           │ - Source code       │                 │
│   │ - Static analysis   │           │ - Static analysis   │                 │
│   └──────────┬──────────┘           └──────────┬──────────┘                 │
│              │                                 │                            │
│              ▼                                 ▼                            │
│   ┌─────────────────────┐           ┌─────────────────────┐                 │
│   │ FINDINGS:           │           │ FINDINGS:           │                 │
│   │                     │           │                     │                 │
│   │ • Missing param doc │           │ • Wrong return type │                 │
│   │ • Incorrect callee  │           │ • Missing exception │                 │
│   │ • Outdated example  │           │ • Vague description │                 │
│   └──────────┬──────────┘           └──────────┬──────────┘                 │
│              │                                 │                            │
│              └────────────────┬────────────────┘                            │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ Agent C validates   │                                  │
│                    │ findings against    │                                  │
│                    │ code and creates    │                                  │
│                    │ correction tickets  │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Strategy 8: Test Generation Validation

**Concept:** Generate unit tests from documentation. If tests fail on actual code, documentation is wrong. If tests pass on buggy code, documentation is incomplete.

```python
class TestGenerationValidator:
    """Validates documentation by generating and running tests."""
    
    async def generate_tests_from_docs(
        self,
        documentation: ComponentDoc
    ) -> List[TestCase]:
        """Generate unit tests based solely on documentation."""
        
        prompt = f"""
        Based on this documentation, generate comprehensive unit tests:
        
        {documentation.to_string()}
        
        Generate tests for:
        1. Normal behavior (happy path)
        2. Each documented exception
        3. Edge cases mentioned
        4. Boundary conditions
        5. Return value validation
        
        Output as pytest-compatible Python code.
        """
        
        test_code = await self.llm.generate(prompt)
        return self._parse_tests(test_code)
    
    async def validate_tests(
        self,
        tests: List[TestCase],
        actual_code: str
    ) -> TestValidationResult:
        """Run generated tests against actual code."""
        
        results = await self._run_tests(tests, actual_code)
        
        failures = [r for r in results if not r.passed]
        
        # Analyze failures
        doc_errors = []
        for failure in failures:
            analysis = await self._analyze_failure(failure, actual_code)
            if analysis.cause == "documentation_wrong":
                doc_errors.append(DocumentationError(
                    test=failure.test_name,
                    expected=failure.expected,
                    actual=failure.actual,
                    recommendation=analysis.fix
                ))
        
        return TestValidationResult(
            total_tests=len(tests),
            passed=len(results) - len(failures),
            failed=len(failures),
            documentation_errors=doc_errors
        )
```

---

## Verification Pipeline Integration

### Complete Flow with All Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ENHANCED VERIFICATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: DOCUMENTATION GENERATION (existing)                              │
│   ────────────────────────────────────────────                              │
│   Stream A ──► Document ──► Validate                                        │
│   Stream B ──► Document ──► Validate                                        │
│                                                                             │
│   PHASE 2: PASSIVE COMPARISON (existing)                                    │
│   ────────────────────────────────────────                                  │
│   Agent C compares A vs B                                                   │
│   Resolves via ground truth                                                 │
│                                                                             │
│   PHASE 3: ACTIVE VERIFICATION (NEW)                                        │
│   ─────────────────────────────────────                                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │  3a. Q&A INTERROGATION                                              │   │
│   │      C generates questions ──► A & B answer ──► Compare to truth    │   │
│   │      └── Identifies: Knowledge gaps, misunderstandings              │   │
│   │                                                                     │   │
│   │  3b. MASKED RECONSTRUCTION                                          │   │
│   │      C masks code ──► A & B reconstruct ──► Score accuracy          │   │
│   │      └── Identifies: Missing specifics, vague documentation         │   │
│   │                                                                     │   │
│   │  3c. SCENARIO WALKTHROUGH                                           │   │
│   │      C provides inputs ──► A & B trace execution ──► Compare        │   │
│   │      └── Identifies: Call graph errors, missing side effects        │   │
│   │                                                                     │   │
│   │  3d. MUTATION DETECTION                                             │   │
│   │      C introduces bugs ──► A & B assess detectability               │   │
│   │      └── Identifies: Imprecise boundary/logic documentation         │   │
│   │                                                                     │   │
│   │  3e. IMPACT ANALYSIS                                                │   │
│   │      C proposes changes ──► A & B predict impact ──► Verify         │   │
│   │      └── Identifies: Missing dependency documentation               │   │
│   │                                                                     │   │
│   │  3f. ADVERSARIAL REVIEW                                             │   │
│   │      A reviews B's docs ◄──► B reviews A's docs                     │   │
│   │      └── Identifies: Cross-team blind spots                         │   │
│   │                                                                     │   │
│   │  3g. TEST GENERATION                                                │   │
│   │      Generate tests from docs ──► Run against code                  │   │
│   │      └── Identifies: Behavioral mismatches                          │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   PHASE 4: CONSOLIDATION                                                    │
│   ──────────────────────                                                    │
│   Aggregate all verification findings                                       │
│   Generate JIRA tickets for gaps                                            │
│   Update documentation                                                      │
│   Re-verify until quality threshold met                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Verification Score Aggregation

```python
@dataclass
class VerificationScores:
    """Aggregated scores from all verification strategies."""
    
    qa_score: float           # % of questions answered correctly
    reconstruction_score: float  # % of masked elements reconstructed
    scenario_score: float     # % of execution traces correct
    mutation_score: float     # % of mutations detectable from docs
    impact_score: float       # % of impacts correctly predicted
    adversarial_findings: int # Number of issues found by cross-review
    test_pass_rate: float     # % of generated tests that pass
    
    @property
    def overall_quality(self) -> float:
        """Weighted overall documentation quality score."""
        weights = {
            "qa": 0.15,
            "reconstruction": 0.20,
            "scenario": 0.20,
            "mutation": 0.15,
            "impact": 0.15,
            "test": 0.15
        }
        
        return (
            self.qa_score * weights["qa"] +
            self.reconstruction_score * weights["reconstruction"] +
            self.scenario_score * weights["scenario"] +
            self.mutation_score * weights["mutation"] +
            self.impact_score * weights["impact"] +
            self.test_pass_rate * weights["test"]
        )
    
    @property
    def quality_grade(self) -> str:
        score = self.overall_quality
        if score >= 0.95:
            return "A"  # Production ready
        elif score >= 0.85:
            return "B"  # Minor gaps
        elif score >= 0.70:
            return "C"  # Significant gaps
        else:
            return "F"  # Major revision needed

    def get_weakest_areas(self) -> List[str]:
        """Identify verification areas needing most improvement."""
        scores = {
            "Q&A Knowledge": self.qa_score,
            "Implementation Details": self.reconstruction_score,
            "Execution Behavior": self.scenario_score,
            "Boundary Precision": self.mutation_score,
            "Dependency Tracking": self.impact_score,
            "Behavioral Accuracy": self.test_pass_rate
        }
        
        return sorted(scores.keys(), key=lambda k: scores[k])[:3]
```

---

## Configuration

```yaml
# verification_config.yaml

verification:
  enabled_strategies:
    - qa_interrogation
    - masked_reconstruction
    - scenario_walkthrough
    - mutation_detection
    - impact_analysis
    - adversarial_review
    - test_generation
  
  thresholds:
    min_overall_quality: 0.85  # Minimum to pass
    min_qa_score: 0.80
    min_reconstruction_score: 0.75
    min_scenario_score: 0.85
    min_test_pass_rate: 0.90
  
  qa_interrogation:
    questions_per_component: 5
    categories:
      - return_value
      - error_handling
      - edge_case
      - call_flow
  
  masked_reconstruction:
    mask_ratio: 0.3
    mask_types:
      - constants
      - conditions
      - returns
  
  scenario_walkthrough:
    scenarios_per_component: 3
    types:
      - happy_path
      - error_path
      - edge_case
  
  mutation_detection:
    mutations_per_component: 5
    mutation_types:
      - boundary
      - off_by_one
      - null_handling
  
  adversarial_review:
    max_findings_per_component: 10
  
  test_generation:
    tests_per_component: 10
    run_generated_tests: true
```

---

## Summary: Verification Strategy Matrix

| Strategy | What It Tests | Catches | Cost | Priority |
|----------|--------------|---------|------|----------|
| **Q&A Interrogation** | Knowledge completeness | Missing info, wrong understanding | Medium | High |
| **Masked Reconstruction** | Implementation detail docs | Vague descriptions, missing specifics | Medium | High |
| **Scenario Walkthrough** | Behavioral accuracy | Wrong call graphs, missing side effects | High | Critical |
| **Mutation Detection** | Precision of conditions | Boundary errors, logic gaps | Medium | Medium |
| **Impact Analysis** | Dependency documentation | Missing callers/callees | Low | High |
| **Adversarial Review** | Cross-validation | Blind spots both teams missed | Medium | Medium |
| **Test Generation** | Behavioral correctness | Semantic errors | High | Critical |
| **Code Reconstruction** | Completeness for rebuild | Gaps that block implementation | High | Critical |

Recommended minimum for production: **Q&A + Scenario + Test Generation**
