from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


chat_model = ChatOllama(model="llama3.2:1b")

prompt1 = PromptTemplate(
    template="Generate a short and simple notes on given text:\n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate quiz of 10 questions on a given text:\n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template=" Merge both provided notes and quiz into single documentation \n notes -> {notes} , quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

notes_chain = prompt1 | chat_model | parser
quiz_chain = prompt2 | chat_model | parser

parllel_chain = RunnableParallel({
    "notes" : notes_chain,
    "quiz" : quiz_chain
})

merge_chain = parllel_chain | prompt3 | chat_model | parser

text = """
lass sklearn.linear_model.LogisticRegression(penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, n_jobs=None)[source]
Logistic Regression (aka logit, MaxEnt) classifier.

This class implements regularized logistic regression using a set of available solvers. Note that regularization is applied by default. It can handle both dense and sparse input X. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied).

The solvers ‘lbfgs’, ‘newton-cg’, ‘newton-cholesky’ and ‘sag’ support only L2 regularization with primal formulation, or no regularization. The ‘liblinear’ solver supports both L1 and L2 regularization (but not both, i.e. elastic-net), with a dual formulation only for the L2 penalty. The Elastic-Net (combination of L1 and L2) regularization is only supported by the ‘saga’ solver.

For multiclass problems (whenever n_classes >= 3), all solvers except ‘liblinear’ optimize the (penalized) multinomial loss. ‘liblinear’ only handles binary classification but can be extended to handle multiclass by using OneVsRestClassifier.

Read more in the User Guide.

Parameters:
penalty{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
Specify the norm of the penalty:

None: no penalty is added;

'l2': add an L2 penalty term and it is the default choice;

'l1': add an L1 penalty term;

'elasticnet': both L1 and L2 penalty terms are added.

Warning

Some penalties may not work with some solvers. See the parameter solver below, to know the compatibility between the penalty and solver.

Added in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)

Deprecated since version 1.8: penalty was deprecated in version 1.8 and will be removed in 1.10. Use l1_ratio and C instead. l1_ratio=0 for penalty='l2', l1_ratio=1 for penalty='l1', l1_ratio set to any float between 0 and 1 for penalty='elasticnet', and C=np.inf for penalty=None.

Cfloat, default=1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. C=np.inf results in unpenalized logistic regression. For a visual example on the effect of tuning the C parameter with an L1 penalty, see: Regularization path of L1- Logistic Regression.

l1_ratiofloat, default=0.0
The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Setting l1_ratio=1 gives a pure L1-penalty, setting l1_ratio=0 a pure L2-penalty. Any value between 0 and 1 gives an Elastic-Net penalty of the form l1_ratio * L1 + (1 - l1_ratio) * L2.

Warning

Certain values of l1_ratio, i.e. some penalties, may not work with some solvers. See the parameter solver below, to know the compatibility between the penalty and solver.

Changed in version 1.8: Default value changed from None to 0.0.

Deprecated since version 1.8: None is deprecated and will be removed in version 1.10. Always use l1_ratio to specify the penalty type.

dualbool, default=False
Dual (constrained) or primal (regularized, see also this equation) formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.

tolfloat, default=1e-4
Tolerance for stopping criteria.

fit_interceptbool, default=True
Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

intercept_scalingfloat, default=1
Useful only when the solver liblinear is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.

Note

The synthetic feature weight is subject to L1 or L2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

class_weightdict or ‘balanced’, default=None
Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.

The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

Added in version 0.17: class_weight=’balanced’

random_stateint, RandomState instance, default=None
Only used for solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. It has no effect on the other solvers. See Glossary for details.

solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:

‘lbfgs’ is a good default solver because it works reasonably well for a wide class of problems.

For multiclass problems (n_classes >= 3), all solvers except ‘liblinear’ minimize the full multinomial loss, ‘liblinear’ will raise an error.

‘newton-cholesky’ is a good choice for n_samples >> n_features * n_classes, especially with one-hot encoded categorical features with rare categories. Be aware that the memory usage of this solver has a quadratic dependency on n_features * n_classes because it explicitly computes the full Hessian matrix.

For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;

‘liblinear’ can only handle binary classification by default. To apply a one-versus-rest scheme for the multiclass setting one can wrap it with the OneVsRestClassifier.

Warning

The choice of the algorithm depends on the penalty chosen (l1_ratio=0 for L2-penalty, l1_ratio=1 for L1-penalty and 0 < l1_ratio < 1 for Elastic-Net) and on (multinomial) multiclass support:

solver

l1_ratio

multinomial multiclass

‘lbfgs’

l1_ratio=0

yes

‘liblinear’

l1_ratio=1 or l1_ratio=0

no

‘newton-cg’

l1_ratio=0

yes

‘newton-cholesky’

l1_ratio=0

yes

‘sag’

l1_ratio=0

yes

‘saga’

0<=l1_ratio<=1

yes

Note

‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

See also

Refer to the User Guide for more information regarding LogisticRegression and more specifically the Table summarizing solver/penalty supports.

Added in version 0.17: Stochastic Average Gradient (SAG) descent solver. Multinomial support in version 0.18.

Added in version 0.19: SAGA solver.

Changed in version 0.22: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22.

Added in version 1.2: newton-cholesky solver. Multinomial support in version 1.6.

max_iterint, default=100
Maximum number of iterations taken for the solvers to converge.

verboseint, default=0
For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

warm_startbool, default=False
When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary.
"""
result = merge_chain.invoke({'text':text})
print(result)


