# Job Shop Scheduling Constraints

### **1. Start Constraints**

#### **Rule 1: A job can only start if it becomes active**
- **Logical**: $ Y_t \implies X_t $
- **Algebraic**: $ Y_t \leq X_t $

---

#### **Rule 2: If the job becomes active now but wasn’t active before, it must have started**
- **Logical**: $ \neg X_{t-1} \land X_t \implies Y_t $
- **Algebraic**: $ X_t - X_{t-1} \leq Y_t $

---

#### **Rule 3: A job can’t start if it was already active before**
- **Logical**: $ Y_t \implies \neg X_{t-1} $
- **Algebraic**: $ Y_t \leq 1 - X_{t-1} $

---

#### **Special Case for $ t = 0 $: If a job is active at the start, it must have started**
- **Logical**: $ X_0 \implies Y_0 $
- **Algebraic**: $ X_0 \leq Y_0 $

---

### **2. End Constraints**

#### **Rule 1: A job can only end if it is active**
- **Logical**: $ Z_t \implies X_t $
- **Algebraic**: $ Z_t \leq X_t $

---

#### **Rule 2: If the job is active now but inactive next, it must have ended**
- **Logical**: $ X_t \land \neg X_{t+1} \implies Z_t $
- **Algebraic**: $ X_t - X_{t+1} \leq Z_t $

---

#### **Rule 3: A job can’t end if it will still be active next**
- **Logical**: $ Z_t \implies \neg X_{t+1} $
- **Algebraic**: $ Z_t \leq 1 - X_{t+1} $

---

#### **Special Case for $ t = \text{end} $: If a job is still active at the final time step, it must end**
- **Logical**: $ X_{\text{end}} \implies Z_{\text{end}} $
- **Algebraic**: $ X_{\text{end}} \leq Z_{\text{end}} $

---

### **Summary Table**

| **Constraint**                                      | **Logical Form**                        | **Algebraic Form**                   |
|-----------------------------------------------------|-----------------------------------------|---------------------------------------|
| **Start Rule 1**: Can only start if active          | $ Y_t \implies X_t $                  | $ Y_t \leq X_t $                    |
| **Start Rule 2**: Active now but not before → start | $ \neg X_{t-1} \land X_t \implies Y_t $ | $ X_t - X_{t-1} \leq Y_t $          |
| **Start Rule 3**: Can’t start if active before      | $ Y_t \implies \neg X_{t-1} $         | $ Y_t \leq 1 - X_{t-1} $            |
| **Start Special Case ($ t = 0 $)**               | $ X_0 \implies Y_0 $                  | $ X_0 \leq Y_0 $                    |
| **End Rule 1**: Can only end if active             | $ Z_t \implies X_t $                  | $ Z_t \leq X_t $                    |
| **End Rule 2**: Active now but not next → end      | $ X_t \land \neg X_{t+1} \implies Z_t $ | $ X_t - X_{t+1} \leq Z_t $          |
| **End Rule 3**: Can’t end if active next           | $ Z_t \implies \neg X_{t+1} $         | $ Z_t \leq 1 - X_{t+1} $            |
| **End Special Case ($ t = \text{end} $)**        | $ X_{\text{end}} \implies Z_{\text{end}} $ | $ X_{\text{end}} \leq Z_{\text{end}} $ |

---

### 3. Duration Constraints

#### **Desired Behavior**
If a job starts at time $ t $, it must remain active for $ p $ consecutive time steps, where $ p $ is the duration of the job. For example:
- If a job starts at $ t = 3 $ and $ p = 4 $, it must be active at $ t = 3, 4, 5, 6 $.

---

### **Framework for Deriving the Constraints**

#### **Step 1: Define Variables**
- $ X_t $: Binary variable indicating if the job is active at time $ t $.
- $ Y_t $: Binary variable indicating if the job starts at time $ t $.
- $ p $: The duration of the job.

#### **Step 2: Describe the Behavior in Natural Language**
1. If the job starts at $ t $ ($ Y_t = 1 $):
   - The job must be active from $ t $ to $ t + p - 1 $ ($ X_t, X_{t+1}, \dots, X_{t+p-1} = 1 $).
2. If the job doesn’t start at $ t $, there’s no requirement imposed for that time interval.

#### **Step 3: Logical Form**
For each time $ t $, if $ Y_t = 1 $, then:
$
X_{t+k} = 1 \quad \forall k \in \{0, 1, \dots, p-1\}
$

#### **Step 4: Algebraic Form**
To enforce this behavior, we write:
$
Y_t \implies X_{t+k}, \quad \forall k \in \{0, 1, \dots, p-1\}
$

Using the implication rule ($ Y_t \leq X_{t+k} $), we derive:
$
X_{t+k} \geq Y_t, \quad \forall k \in \{0, 1, \dots, p-1\}
$

---

### **Combined Constraints for All $ t $**
For all time steps $ t $ and durations $ k \in \{0, 1, \dots, p-1\}$:
$
X_{t+k} \geq Y_t
$

---

### **Example**

#### **Scenario**
- $ T = 10 $: The total time horizon is 10.
- $ p = 3 $: The job duration is 3.

#### **If $ Y_4 = 1 $:**
- The job must be active at $ t = 4, 5, 6 $.

**Constraints:**
$
X_4 \geq Y_4, \quad X_5 \geq Y_4, \quad X_6 \geq Y_4
$

---

### **Compact Formulation Using Summation**
To compactly enforce the constraints for all $ t $, you can write:
$
X_{t+k} \geq Y_t, \quad \forall t, k \in \{0, \dots, p-1\}
$

If you’re summing over valid $ t + k $ ranges:
$
\sum_{k=0}^{p-1} X_{t+k} \geq p \cdot Y_t
$

---

### **Summary**

1. **Natural Language**: If a job starts at time $ t $, it must remain active for $ p $ consecutive time steps.
2. **Logical Form**: $ Y_t \implies X_{t+k}, \forall k \in \{0, 1, \dots, p-1\} $.
3. **Algebraic Constraints**:
   - $ X_{t+k} \geq Y_t, \forall k \in \{0, 1, \dots, p-1\} $, applied within the time horizon.
4. **Compact Formulation**: Consider the cumulative sum of active states over the duration $ p $:
$
\sum_{k=0}^{p-1} X_{t+k} \geq p \cdot Y_t
$ . <br> This ensures that if a job starts at time $ t $, the sum of active states over the next $ p $ time steps must be at least $ p $, enforcing continuous activity.
5. **Boundary Condition**: Ensure $ t + k \leq T $ when formulating the constraints.



