## Merge Request Summary
(to be completed by merge request author)

- Reason for the merge request (new feature/bugfix/refactoring)?

- Briefly explain why the new code or changes are necessary and how it is implemented?

- [ ] Assign a reviewer for this merge request!

## Code-Review Checklist
(for assigned code reviewer)

The assigned reviewer should look at each item in this checklist. For anything that 
needs to be addressed or that can be improved, add comment (if necessary including 
a ToDo-Checklist).

Once the checklist is completed, add an "Assignee" who can merge the code.

- [ ] Functionality
    - Does the code implement the intended functionality?
    - Are edge cases and potential error scenarios handled or documented appropriately?

- [ ] Test Coverage
    - Does the new code include appropriate unit tests or integration tests?
    - Are the tests passing and up-to-date?
    - Is the test coverage sufficient for the critical functionality and edge cases?
 
- [ ] Code/Developer Documentation
    - Are inline comments used effectively to explain complex or non-obvious code segments (docstring coverage)?
    - Do functions, methods, and classes have descriptive comments or docstrings?
    - Do these comments/docstrings show the formulas/equations/algorithms used with dimensions and units (possibly with reference to literature)?
    - Are all in/output variables labeled with meaning and data type?
    - Do the variables in the code have meaningful and clear names?
    - Do the docstrings have a suitable formatting (Numpydoc)?
      https://numpydoc.readthedocs.io/en/latest/format.html.

- [ ] User/high-level Documentation
    - Is there high-level documentation for complex modules or components?
    - Code-Architecture diagrams?
    - Are there any application examples?
    - User Manuals?

- [ ] Readability and Maintainability
    - Is the code well-organized and easy to read ?
    - Are naming conventions consistent and descriptive?
    - Is the code properly indented and formatted (flake8/pylint)?
 
- [ ] Code Structure and Design
    - Is the code modular and maintainable (spaghetti code)?
    - Are functions and classes of reasonable size and complexity?
    - Does the code adhere to the principles of separation of concerns and single responsibility (one function = one task)?
 
- [ ] Reuse and Dependencies
    - Is the code properly reusing existing libraries, frameworks, or components?
    - Are dependencies managed correctly and up-to-date?
    - Are any unnecessary dependencies or duplicate code segments removed?
 
- [ ] Performance and Efficiency (optional)
    - Are there any potential performance bottlenecks or inefficiencies?

## Merge Checklist

(for assignee who is finally merging this)

- [ ] Do all CI pipelines pass?
- [ ] Has the code review been completed?
- [ ] Are all authors included in the PyADF author list?
