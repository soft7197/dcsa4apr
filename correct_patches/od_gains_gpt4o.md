# Overfitting Detector Gains - GPT-4o

Overfitting detector gain bugs: 15

These bugs had **no correct patches before OD refinement**. All pre-refinement patches were overfitting ??they passed tests but did not actually fix the root cause. OD refinement produced correct patches for these bugs.

---

## Chart-5

**Root cause:** When duplicate X values are allowed, addOrUpdate still uses indexOf/binary-search-style insertion logic, so an existing X can yield a non-negative index that is incorrectly converted into an insertion position, causing wrong behavior or IndexOutOfBoundsException instead of simply adding a new item.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test by avoiding the invalid -index-1 computation when duplicates are allowed, but it does not fully fix the method's intended behavior. For an auto-sorted series with allowDuplicateXValues=false, adding a new X not already present gives index < 0 and should insert at -index-1; the pre-patch instead appends, producing an unsorted series and thus incorrect behavior on reasonable inputs beyond the test.

**OD refinement (NECESSARY):** The refinement changed the insertion logic in the else-branch from appending whenever duplicates are allowed to computing a valid sorted insertion index whenever autoSort is enabled and index < 0, otherwise appending at the end. This is necessary because the pre-patch breaks sorted-series semantics for new non-duplicate X values when allowDuplicateXValues is false: it appends instead of inserting at the proper sorted position.

**Verdict:** Because the OD refinement introduced necessary missing logic for correct insertion in auto-sorted, non-duplicate series, the pre-patch was not semantically complete. It overfits to the observed duplicate-allowed failure while regressing or preserving incorrect behavior for other valid cases.

**Before OD (overfitting):**
```diff
@@ -4,5 +4,4 @@
 }
 
-// if we get to here, we know that duplicate X values are not permitted
 XYDataItem overwritten = null;
 int index = indexOf(x);
@@ -16,17 +15,10 @@
 }
 existing.setY(y);
-}
-else {
-// if the series is sorted, the negative index is a result from
-// Collections.binarySearch() and tells us where to insert the
-// new item...otherwise it will be just -1 and we should just
-// append the value to the list...
-if (this.autoSort) {
+} else {
+if (this.autoSort && !this.allowDuplicateXValues) {
 this.data.add(-index - 1, new XYDataItem(x, y));
-}
-else {
+} else {
 this.data.add(new XYDataItem(x, y));
 }
-// check if this addition will exceed the maximum item count...
 if (getItemCount() > this.maximumItemCount) {
 this.data.remove(0);
```

**After OD (correct):**
```diff
@@ -4,5 +4,4 @@
 }
 
-// if we get to here, we know that duplicate X values are not permitted
 XYDataItem overwritten = null;
 int index = indexOf(x);
@@ -11,22 +10,12 @@
 try {
 overwritten = (XYDataItem) existing.clone();
-}
-catch (CloneNotSupportedException e) {
+} catch (CloneNotSupportedException e) {
 throw new SeriesException("Couldn't clone XYDataItem!");
 }
 existing.setY(y);
-}
-else {
-// if the series is sorted, the negative index is a result from
-// Collections.binarySearch() and tells us where to insert the
-// new item...otherwise it will be just -1 and we should just
-// append the value to the list...
-if (this.autoSort) {
-this.data.add(-index - 1, new XYDataItem(x, y));
-}
-else {
-this.data.add(new XYDataItem(x, y));
-}
-// check if this addition will exceed the maximum item count...
+} else {
+int insertIndex = (this.autoSort && index < 0) ? -index - 1 : this.data.size();
+this.data.add(insertIndex, new XYDataItem(x, y));
+
 if (getItemCount() > this.maximumItemCount) {
 this.data.remove(0);
```

**Ground truth fix:**
```diff
@@ -2,4 +2,8 @@
 if (x == null) {
 throw new IllegalArgumentException("Null 'x' argument.");
+}
+if (this.allowDuplicateXValues) {
+add(x, y);
+return null;
 }
 
@@ -7,5 +11,5 @@
 XYDataItem overwritten = null;
 int index = indexOf(x);
-if (index >= 0 && !this.allowDuplicateXValues) {
+if (index >= 0) {
 XYDataItem existing = (XYDataItem) this.data.get(index);
 try {
```

---

## Cli-20

**Root cause:** The bug is that long tokens beginning with "--" and containing '=' were always split into option and value, even when the long option was unrecognized and stopAtNonOption should have treated the whole token as a non-option and stopped option parsing.

**Why pre-patch overfits:** The pre-patch fixes the specific failing scenario for unknown long options that contain '=', but it does not fully fix the root cause for long options in general. On reasonable inputs such as an unrecognized plain long option like "--zop" with stopAtNonOption=true, it would still mishandle parsing by not treating it as a non-option and not stopping, unlike the ground-truth/refined behavior.

**OD refinement (NECESSARY):** The refinement changed the no-'=' long-option case to also validate whether the long option itself is recognized before blindly adding it as an option token. In the pre-patch, only "--foo=bar" checked options.hasOption(name); plain "--foo" was always added as a token, so an unrecognized long option with stopAtNonOption=true would not trigger process(token) and stop parsing as required.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic refactor, the pre-patch was still semantically incomplete. It overfit to the '=' case from the tests and failed to implement the full long-option/non-option handling required by the correct fix.

**Before OD (overfitting):**
```diff
@@ -1,4 +1,3 @@
-protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption)
-{
+protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption) {
 init();
 this.options = options;
@@ -8,19 +7,21 @@
 
 // process each command line token
-while (iter.hasNext())
-{
+while (iter.hasNext()) {
 // get the next command line token
 String token = (String) iter.next();
 
 // handle long option --foo or --foo=bar
-if (token.startsWith("--"))
-{
-if (token.indexOf('=') != -1)
-{
-tokens.add(token.substring(0, token.indexOf('=')));
+if (token.startsWith("--")) {
+if (token.indexOf('=') != -1) {
+String name = token.substring(0, token.indexOf('='));
+if (options.hasOption(name)) {
+tokens.add(name);
 tokens.add(token.substring(token.indexOf('=') + 1, token.length()));
+} else if (stopAtNonOption) {
+process(token);
+} else {
+tokens.add(token);
 }
-else
-{
+} else {
 tokens.add(token);
 }
@@ -28,30 +29,19 @@
 
 // single hyphen
-else if ("-".equals(token))
-{
+else if ("-".equals(token)) {
 tokens.add(token);
-}
-else if (token.startsWith("-"))
-{
-if (token.length() == 2)
-{
+} else if (token.startsWith("-")) {
+if (token.length() == 2) {
 processOptionToken(token, stopAtNonOption);
-}
-else if (options.hasOption(token))
-{
+} else if (options.hasOption(token)) {
 tokens.add(token);
 }
 // requires bursting
-else
-{
+else {
 burstToken(token, stopAtNonOption);
 }
-}
-else if (stopAtNonOption)
-{
+} else if (stopAtNonOption) {
 process(token);
-}
-else
-{
+} else {
 tokens.add(token);
 }
```

**After OD (correct):**
```diff
@@ -1,4 +1,3 @@
-protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption)
-{
+protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption) {
 init();
 this.options = options;
@@ -8,19 +7,21 @@
 
 // process each command line token
-while (iter.hasNext())
-{
+while (iter.hasNext()) {
 // get the next command line token
 String token = (String) iter.next();
 
 // handle long option --foo or --foo=bar
-if (token.startsWith("--"))
-{
-if (token.indexOf('=') != -1)
-{
-tokens.add(token.substring(0, token.indexOf('=')));
-tokens.add(token.substring(token.indexOf('=') + 1, token.length()));
+if (token.startsWith("--")) {
+int equalIndex = token.indexOf('=');
+String optionName = (equalIndex != -1) ? token.substring(0, equalIndex) : token;
+
+if (options.hasOption(optionName)) {
+tokens.add(optionName);
+if (equalIndex != -1) {
+tokens.add(token.substring(equalIndex + 1));
 }
-else
-{
+} else if (stopAtNonOption) {
+process(token);
+} else {
 tokens.add(token);
 }
@@ -28,30 +29,19 @@
 
 // single hyphen
-else if ("-".equals(token))
-{
+else if ("-".equals(token)) {
 tokens.add(token);
-}
-else if (token.startsWith("-"))
-{
-if (token.length() == 2)
-{
+} else if (token.startsWith("-")) {
+if (token.length() == 2) {
 processOptionToken(token, stopAtNonOption);
-}
-else if (options.hasOption(token))
-{
+} else if (options.hasOption(token)) {
 tokens.add(token);
 }
 // requires bursting
-else
-{
+else {
 burstToken(token, stopAtNonOption);
 }
-}
-else if (stopAtNonOption)
-{
+} else if (stopAtNonOption) {
 process(token);
-}
-else
-{
+} else {
 tokens.add(token);
 }
```

**Ground truth fix:**
```diff
@@ -16,12 +16,17 @@
 if (token.startsWith("--"))
 {
-if (token.indexOf('=') != -1)
+int pos = token.indexOf('=');
+String opt = pos == -1 ? token : token.substring(0, pos); // --foo
+
+if (!options.hasOption(opt) && stopAtNonOption)
 {
-tokens.add(token.substring(0, token.indexOf('=')));
-tokens.add(token.substring(token.indexOf('=') + 1, token.length()));
+process(token);
 }
 else
 {
-tokens.add(token);
+tokens.add(opt);
+if (pos != -1) {
+tokens.add(token.substring(pos + 1));
+}
 }
 }
```

---

## Cli-38

**Root cause:** isShortOption incorrectly required the entire post-dash token to be a declared short option, instead of recognizing that a token can start with a valid short option and then contain concatenated options or attached value text.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause: it rejects valid short options such as '-t1' when 't1' is itself a defined short option, and it also mishandles forms with '=' because it iterates over all characters including non-option/value characters. Thus it would fail on reasonable inputs beyond the triggering test and is an overfitting patch.

**OD refinement (NECESSARY):** The refinement changed the logic from validating every character after '-' as an independent short option to extracting the option name before '=' and accepting the token if its first character is a valid short option. This is necessary because short options in Commons CLI may be multi-character (like t1) or may have attached values, so per-character validation is semantically too strict.

**Verdict:** Because the OD refinement introduced necessary missing logic rather than a cosmetic rewrite, and the pre-patch's per-character check is semantically wrong for valid multi-character short options and attached values, the original patch was a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,19 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+if (token.charAt(1) == '-') {
+// Handle '--option' as long option, not short
+return false;
 }
+
+// remove leading "-"
+String reducedToken = token.substring(1);
+
+// check each character in reducedToken for short options
+for (char opt : reducedToken.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(opt))) {
+return false;
+}
+}
+
+return true;
+}
```

**After OD (correct):**
```diff
@@ -7,8 +7,15 @@
 }
 
-// remove leading "-" and "=value"
+if (token.charAt(1) == '-') {
+// Handle '--option' as long option, not short
+return false;
+}
+
+// remove leading "-" and extract option name before "="
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// check the first character of reduced token to see if it's a short option
+// if the first char is valid, assume others are values or part of option name
+return optName.length() > 0 && options.hasShortOption(String.valueOf(optName.charAt(0)));
 }
```

**Ground truth fix:**
```diff
@@ -10,5 +10,9 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
+if (options.hasShortOption(optName))
+{
+return true;
+}
 // check for several concatenated short options
+return optName.length() > 0 && options.hasShortOption(String.valueOf(optName.charAt(0)));
 }
```

---

## Closure-66

**Root cause:** The bug is that STRING nodes used as object literal keys were still counted as typeable in typed-percentage accounting, even though object literal keys should be excluded from that metric.

**Why pre-patch overfits:** The pre-patch fixes the failing tests by excluding object literal keys, but it does so too broadly: it makes every STRING node non-typeable, so normal string literals would stop contributing to typed-percentage accounting. It also incorrectly makes OBJECTLIT nodes non-typeable, which would undercount typed expressions on many reasonable inputs beyond the provided tests.

**OD refinement (NECESSARY):** The OD refinement changed the STRING case from unconditionally setting typeable = false to setting it false only when the STRING node is an object literal key, while keeping ordinary string literals typeable. It also removed the pre-patch's added typeable = false for OBJECTLIT/AND/HOOK/OR, restoring correct accounting for those expression nodes. These are semantic changes, not cosmetic ones.

**Verdict:** Because the refinement introduced necessary logic that the pre-patch lacked, the original patch was overfitting. Its unconditional exclusion of STRING nodes and OBJECTLIT-family nodes changes semantics beyond the root-cause fix and would yield incorrect typed-percentage results on other programs.

**Before OD (overfitting):**
```diff
@@ -3,5 +3,4 @@
 JSType leftType, rightType;
 Node left, right;
-// To be explicitly set to false if the node is not typeable.
 boolean typeable = true;
 
@@ -12,6 +11,4 @@
 
 case Token.LP:
-// If this is under a FUNCTION node, it is a parameter list and can be
-// ignored here.
 if (parent.getType() != Token.FUNCTION) {
 ensureTyped(t, n, getJSType(n.getFirstChild()));
@@ -51,14 +48,12 @@
 
 case Token.STRING:
-// Object literal keys are handled with OBJECTLIT
 if (!NodeUtil.isObjectLitKey(n, n.getParent())) {
 ensureTyped(t, n, STRING_TYPE);
-// Object literal keys are not typeable
-}
+}
+typeable = false;
 break;
 
 case Token.GET:
 case Token.SET:
-// Object literal keys are handled with OBJECTLIT
 break;
 
@@ -79,7 +74,4 @@
 case Token.GETELEM:
 visitGetElem(t, n);
-// The type of GETELEM is always unknown, so no point counting that.
-// If that unknown leaks elsewhere (say by an assignment to another
-// variable), then it will be counted.
 typeable = false;
 break;
@@ -191,9 +183,5 @@
 } else if (leftType.matchesNumberContext() &&
 rightType.matchesNumberContext()) {
-// OK.
 } else {
-// Whether the comparison is numeric will be determined at runtime
-// each time the expression is evaluated. Regardless, both operands
-// should match a string context.
 String message = "left side of comparison";
 validator.expectString(t, n, leftType, message);
@@ -288,5 +276,4 @@
 break;
 
-// These nodes have no interesting type behavior.
 case Token.LABEL:
 case Token.LABEL_NAME:
@@ -306,5 +293,4 @@
 break;
 
-// These nodes require data flow analysis.
 case Token.DO:
 case Token.FOR:
@@ -314,13 +300,11 @@
 break;
 
-// These nodes are typed during the type inference.
 case Token.AND:
 case Token.HOOK:
 case Token.OBJECTLIT:
 case Token.OR:
-if (n.getJSType() != null) { // If we didn't run type inference.
+if (n.getJSType() != null) {
 ensureTyped(t, n);
 } else {
-// If this is an enum, then give that type to the objectlit as well.
 if ((n.getType() == Token.OBJECTLIT)
 && (parent.getJSType() instanceof EnumType)) {
@@ -335,4 +319,5 @@
 }
 }
+typeable = false;
 break;
 
@@ -343,5 +328,4 @@
 }
 
-// Don't count externs since the user's code may not even use that part.
 typeable = typeable && !inExterns;
 
```

**After OD (correct):**
```diff
@@ -3,5 +3,4 @@
 JSType leftType, rightType;
 Node left, right;
-// To be explicitly set to false if the node is not typeable.
 boolean typeable = true;
 
@@ -12,6 +11,4 @@
 
 case Token.LP:
-// If this is under a FUNCTION node, it is a parameter list and can be
-// ignored here.
 if (parent.getType() != Token.FUNCTION) {
 ensureTyped(t, n, getJSType(n.getFirstChild()));
@@ -51,8 +48,9 @@
 
 case Token.STRING:
-// Object literal keys are handled with OBJECTLIT
 if (!NodeUtil.isObjectLitKey(n, n.getParent())) {
 ensureTyped(t, n, STRING_TYPE);
-// Object literal keys are not typeable
+typeable = true;
+} else {
+typeable = false; // Object literal keys are not typeable
 }
 break;
@@ -60,5 +58,4 @@
 case Token.GET:
 case Token.SET:
-// Object literal keys are handled with OBJECTLIT
 break;
 
@@ -73,13 +70,9 @@
 case Token.GETPROP:
 visitGetProp(t, n, parent);
-typeable = !(parent.getType() == Token.ASSIGN &&
-parent.getFirstChild() == n);
+typeable = !(parent.getType() == Token.ASSIGN && parent.getFirstChild() == n);
 break;
 
 case Token.GETELEM:
 visitGetElem(t, n);
-// The type of GETELEM is always unknown, so no point counting that.
-// If that unknown leaks elsewhere (say by an assignment to another
-// variable), then it will be counted.
 typeable = false;
 break;
@@ -108,6 +101,5 @@
 case Token.INC:
 left = n.getFirstChild();
-validator.expectNumber(
-t, left, getJSType(left), "increment/decrement");
+validator.expectNumber(t, left, getJSType(left), "increment/decrement");
 ensureTyped(t, n, NUMBER_TYPE);
 break;
@@ -128,6 +120,5 @@
 childType = getJSType(n.getFirstChild());
 if (!childType.matchesInt32Context()) {
-report(t, n, BIT_OPERATION, NodeUtil.opToStr(n.getType()),
-childType.toString());
+report(t, n, BIT_OPERATION, NodeUtil.opToStr(n.getType()), childType.toString());
 }
 ensureTyped(t, n, NUMBER_TYPE);
@@ -148,12 +139,10 @@
 JSType leftTypeRestricted = leftType.restrictByNotNullOrUndefined();
 JSType rightTypeRestricted = rightType.restrictByNotNullOrUndefined();
-TernaryValue result =
-leftTypeRestricted.testForEquality(rightTypeRestricted);
+TernaryValue result = leftTypeRestricted.testForEquality(rightTypeRestricted);
 if (result != TernaryValue.UNKNOWN) {
 if (n.getType() == Token.NE) {
 result = result.not();
 }
-report(t, n, DETERMINISTIC_TEST, leftType.toString(),
-rightType.toString(), result.toString());
+report(t, n, DETERMINISTIC_TEST, leftType.toString(), rightType.toString(), result.toString());
 }
 ensureTyped(t, n, BOOLEAN_TYPE);
@@ -168,8 +157,6 @@
 JSType leftTypeRestricted = leftType.restrictByNotNullOrUndefined();
 JSType rightTypeRestricted = rightType.restrictByNotNullOrUndefined();
-if (!leftTypeRestricted.canTestForShallowEqualityWith(
-rightTypeRestricted)) {
-report(t, n, DETERMINISTIC_TEST_NO_RESULT, leftType.toString(),
-rightType.toString());
+if (!leftTypeRestricted.canTestForShallowEqualityWith(rightTypeRestricted)) {
+report(t, n, DETERMINISTIC_TEST_NO_RESULT, leftType.toString(), rightType.toString());
 }
 ensureTyped(t, n, BOOLEAN_TYPE);
@@ -184,24 +171,15 @@
 rightType = getJSType(n.getLastChild());
 if (rightType.isNumber()) {
-validator.expectNumber(
-t, n, leftType, "left side of numeric comparison");
+validator.expectNumber(t, n, leftType, "left side of numeric comparison");
 } else if (leftType.isNumber()) {
-validator.expectNumber(
-t, n, rightType, "right side of numeric comparison");
-} else if (leftType.matchesNumberContext() &&
-rightType.matchesNumberContext()) {
-// OK.
-} else {
-// Whether the comparison is numeric will be determined at runtime
-// each time the expression is evaluated. Regardless, both operands
-// should match a string context.
+validator.expectNumber(t, n, rightType, "right side of numeric comparison");
+} else if (leftType.matchesNumberContext() && rightType.matchesNumberContext()) {
+} else {
 String message = "left side of comparison";
 validator.expectString(t, n, leftType, message);
-validator.expectNotNullOrUndefined(
-t, n, leftType, message, getNativeType(STRING_TYPE));
+validator.expectNotNullOrUndefined(t, n, leftType, message, getNativeType(STRING_TYPE));
 message = "right side of comparison";
 validator.expectString(t, n, rightType, message);
-validator.expectNotNullOrUndefined(
-t, n, rightType, message, getNativeType(STRING_TYPE));
+validator.expectNotNullOrUndefined(t, n, rightType, message, getNativeType(STRING_TYPE));
 }
 ensureTyped(t, n, BOOLEAN_TYPE);
@@ -224,8 +202,6 @@
 rightType = getJSType(right).restrictByNotNullOrUndefined();
 
-validator.expectAnyObject(
-t, left, leftType, "deterministic instanceof yields false");
-validator.expectActualObject(
-t, right, rightType, "instanceof requires an object");
+validator.expectAnyObject(t, left, leftType, "deterministic instanceof yields false");
+validator.expectActualObject(t, right, rightType, "instanceof requires an object");
 ensureTyped(t, n, BOOLEAN_TYPE);
 break;
@@ -278,6 +254,5 @@
 Node child = n.getFirstChild();
 childType = getJSType(child);
-validator.expectObject(
-t, child, childType, "with requires an object");
+validator.expectObject(t, child, childType, "with requires an object");
 typeable = false;
 break;
@@ -288,5 +263,4 @@
 break;
 
-// These nodes have no interesting type behavior.
 case Token.LABEL:
 case Token.LABEL_NAME:
@@ -306,5 +280,4 @@
 break;
 
-// These nodes require data flow analysis.
 case Token.DO:
 case Token.FOR:
@@ -314,13 +287,11 @@
 break;
 
-// These nodes are typed during the type inference.
 case Token.AND:
 case Token.HOOK:
 case Token.OBJECTLIT:
 case Token.OR:
-if (n.getJSType() != null) { // If we didn't run type inference.
+if (n.getJSType() != null) {
 ensureTyped(t, n);
 } else {
-// If this is an enum, then give that type to the objectlit as well.
 if ((n.getType() == Token.OBJECTLIT)
 && (parent.getJSType() instanceof EnumType)) {
@@ -343,5 +314,4 @@
 }
 
-// Don't count externs since the user's code may not even use that part.
 typeable = typeable && !inExterns;
 
```

**Ground truth fix:**
```diff
@@ -54,5 +54,7 @@
 if (!NodeUtil.isObjectLitKey(n, n.getParent())) {
 ensureTyped(t, n, STRING_TYPE);
+} else {
 // Object literal keys are not typeable
+typeable = false;
 }
 break;
```

---

## Codec-6

**Root cause:** The method can return 0 from base64.readResults() even when InputStream.read(byte[]) must keep trying until it can return decoded bytes or EOF, so it needs to loop while the decoded result length is 0.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it may return bytesRead, which is the number of encoded bytes read from the wrapped stream, not the number of decoded bytes placed into the caller buffer. It also calls base64.readResults() multiple times per invocation, so it can consume available decoded data in the loop condition and then return a later result, leading to wrong counts or spurious EOF on reasonable inputs beyond the triggering test.

**OD refinement (NECESSARY):** The refinement removes the pre-patch's incorrect use of the raw underlying stream byte count as the return value and avoids consuming readResults() inside the loop condition and then calling it again. It changes the logic to repeatedly check decoded output first, read more input only when needed, and return only actual decoded bytes or EOF, which is required for correct InputStream semantics.

**Verdict:** Because the refinement change is NECESSARY and corrects real semantic errors in the pre-patch's return-value and readResults handling, the original patch was overfitting rather than a genuinely correct alternative fix.

**Before OD (overfitting):**
```diff
@@ -9,26 +9,12 @@
 return 0;
 } else {
-/*
-Rationale for while-loop on (readLen == 0):
------
-Base64.readResults() usually returns > 0 or EOF (-1).  In the
-rare case where it returns 0, we just keep trying.
-
-This is essentially an undocumented contract for InputStream
-implementors that want their code to work properly with
-java.io.InputStreamReader, since the latter hates it when
-InputStream.read(byte[]) returns a zero.  Unfortunately our
-readResults() call must return 0 if a large amount of the data
-being decoded was non-base64, so this while-loop enables proper
-interop with InputStreamReader for that scenario.
------
-This is a fix for CODEC-101
-*/
+int bytesRead = 0;
+do {
 if (!base64.hasData()) {
 byte[] buf = new byte[doEncode ? 4096 : 8192];
 int c = in.read(buf);
-// A little optimization to avoid System.arraycopy()
-// when possible.
-if (c > 0 && b.length == len) {
+if (c > 0) {
+bytesRead = c; // Update bytesRead since data was read
+if (b.length == len) {
 base64.setInitialBuffer(b, offset, len);
 }
@@ -38,6 +24,11 @@
 base64.decode(buf, 0, c);
 }
-}
-return base64.readResults(b, offset, len);
+} else if (c == -1) {
+break; // EOF reached
 }
 }
+} while (base64.readResults(b, offset, len) == 0);
+int readResult = base64.readResults(b, offset, len);
+return readResult > 0 ? readResult : (bytesRead > 0 ? bytesRead : -1);
+}
+}
```

**After OD (correct):**
```diff
@@ -9,26 +9,23 @@
 return 0;
 } else {
-/*
-Rationale for while-loop on (readLen == 0):
------
-Base64.readResults() usually returns > 0 or EOF (-1).  In the
-rare case where it returns 0, we just keep trying.
-
-This is essentially an undocumented contract for InputStream
-implementors that want their code to work properly with
-java.io.InputStreamReader, since the latter hates it when
-InputStream.read(byte[]) returns a zero.  Unfortunately our
-readResults() call must return 0 if a large amount of the data
-being decoded was non-base64, so this while-loop enables proper
-interop with InputStreamReader for that scenario.
------
-This is a fix for CODEC-101
-*/
+int bytesRead = 0;
+while (true) {
+int result = base64.readResults(b, offset, len);
+if (result > 0) {
+return result;
+}
 if (!base64.hasData()) {
 byte[] buf = new byte[doEncode ? 4096 : 8192];
 int c = in.read(buf);
-// A little optimization to avoid System.arraycopy()
-// when possible.
-if (c > 0 && b.length == len) {
+if (c == -1) {
+if (bytesRead > 0) {
+return bytesRead;
+} else {
+return -1; // EOF
+}
+}
+if (c > 0) {
+bytesRead += c;
+if (b.length == len) {
 base64.setInitialBuffer(b, offset, len);
 }
@@ -39,5 +36,6 @@
 }
 }
-return base64.readResults(b, offset, len);
 }
 }
+}
+}
```

**Ground truth fix:**
```diff
@@ -9,4 +9,5 @@
 return 0;
 } else {
+int readLen = 0;
 /*
 Rationale for while-loop on (readLen == 0):
@@ -25,4 +26,5 @@
 This is a fix for CODEC-101
 */
+while (readLen == 0) {
 if (!base64.hasData()) {
 byte[] buf = new byte[doEncode ? 4096 : 8192];
@@ -39,5 +41,7 @@
 }
 }
-return base64.readResults(b, offset, len);
+readLen = base64.readResults(b, offset, len);
+}
+return readLen;
 }
 }
```

---

## Compress-28

### Patch 1

**Root cause:** The method mishandles unexpected EOF while reading an entry by treating a premature -1 from the underlying stream as normal end-of-entry instead of throwing an IOException for a truncated TAR entry.

**Why pre-patch overfits:** The pre-patch partially addresses the root cause by throwing on -1, but it adds incorrect logic: it throws whenever totalRead < numToRead. Since short reads are legal for InputStream even when more data remains, the pre-patch can fail on valid TAR streams depending on the underlying stream behavior, so it is not a semantically correct general fix.

**OD refinement (NECESSARY):** The refinement removed the extra check that threw an IOException whenever a read returned fewer bytes than requested, and limited truncation detection to the true EOF case (totalRead == -1). That change is necessary because InputStream.read is allowed to return a short positive read without indicating truncation, so the pre-patch would incorrectly reject valid archives on reasonable inputs.

**Verdict:** Because the refinement change is NECESSARY and fixes a real semantic bug in the pre-patch's truncation detection, the original patch was overfitting rather than already correct.

**Before OD (overfitting):**
```diff
@@ -13,10 +13,14 @@
 
 totalRead = is.read(buf, offset, numToRead);
-count(totalRead);
-
 if (totalRead == -1) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated tar entry");
+}
+
+count(totalRead);
 entryOffset += totalRead;
+
+if (totalRead < numToRead) {
+throw new IOException("Truncated tar entry");
 }
 
```

**After OD (correct):**
```diff
@@ -10,14 +10,14 @@
 }
 
-numToRead = Math.min(numToRead, available());
-
-totalRead = is.read(buf, offset, numToRead);
-count(totalRead);
+int availableToRead = Math.min(numToRead, available());
+totalRead = is.read(buf, offset, availableToRead);
 
 if (totalRead == -1) {
 hasHitEOF = true;
-} else {
+throw new IOException("Unexpected EOF in tar archive");
+}
+
+count(totalRead);
 entryOffset += totalRead;
-}
 
 return totalRead;
```

**Ground truth fix:**
```diff
@@ -13,9 +13,12 @@
 
 totalRead = is.read(buf, offset, numToRead);
-count(totalRead);
 
 if (totalRead == -1) {
+if (numToRead > 0) {
+throw new IOException("Truncated TAR archive");
+}
 hasHitEOF = true;
 } else {
+count(totalRead);
 entryOffset += totalRead;
 }
```

### Patch 2

**Root cause:** The bug is that reaching underlying stream EOF before consuming the full declared tar entry size is treated as normal EOF instead of throwing an IOException for a truncated archive.

**Why pre-patch overfits:** The pre-patch does trigger the failing test, but it does so with overly broad logic that treats any partial read as a truncated TAR entry. On reasonable inputs where the wrapped stream legally returns fewer bytes than requested without EOF, the pre-patch would throw IOException incorrectly, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement changes truncation detection from 'any short read (totalRead < numToRead)' to the correct condition: only throw when the underlying stream returns -1 before the entry's expected size has been fully read. This is necessary because short positive reads are valid for InputStream.read and do not imply truncation, while the pre-patch would incorrectly reject such cases.

**Verdict:** Because the OD refinement introduced necessary semantic logic for distinguishing valid short reads from premature EOF, the pre-patch was not already correct. Its truncation check was overfitting to the test by conflating partial reads with archive truncation.

**Before OD (overfitting):**
```diff
@@ -21,4 +21,8 @@
 }
 
+if (totalRead < numToRead) {
+throw new IOException("Truncated TAR archive entry");
+}
+
 return totalRead;
 }
```

**After OD (correct):**
```diff
@@ -11,13 +11,16 @@
 
 numToRead = Math.min(numToRead, available());
-
 totalRead = is.read(buf, offset, numToRead);
-count(totalRead);
 
 if (totalRead == -1) {
 hasHitEOF = true;
-} else {
+if (entryOffset != entrySize) { // Check if EOF is premature
+throw new IOException("Truncated TAR archive entry");
+}
+return -1;
+}
+
 entryOffset += totalRead;
-}
+count(totalRead);
 
 return totalRead;
```

**Ground truth fix:**
```diff
@@ -13,9 +13,12 @@
 
 totalRead = is.read(buf, offset, numToRead);
-count(totalRead);
 
 if (totalRead == -1) {
+if (numToRead > 0) {
+throw new IOException("Truncated TAR archive");
+}
 hasHitEOF = true;
 } else {
+count(totalRead);
 entryOffset += totalRead;
 }
```

---

## JacksonDatabind-110

**Root cause:** The bug is that deserialization only recognizes one unmodifiable-list implementation class, but `Collections.unmodifiableList(...)` can produce an alias implementation for non-random-access lists that also needs the same converter.

**Why pre-patch overfits:** The pre-patch does not fix the real root cause: it still only matches the exact `UnmodifiableList` class name, so it will not handle the alternate unmodifiable list implementation created from a `LinkedList` (`UnmodifiableRandomAccessList` vs alias counterpart issue). Its added `isTypeOrSubTypeOf(List.class)` guard is irrelevant to the missing case, so the patch remains incomplete and would fail on reasonable inputs beyond the specific matched class.

**OD refinement (NECESSARY):** The refinement broadened detection from a single exact class name (`java.util.Collections$UnmodifiableList`) to a more general unmodifiable-collection check, allowing additional unmodifiable wrapper implementations to be handled. This matters because the pre-patch still misses the alias unmodifiable list class that is the actual source of the bug, whereas the refined logic covers it.

**Verdict:** Because the refinement change is NECESSARY and adds missing semantic coverage for additional unmodifiable collection implementations, the pre-patch was an overfitting/incomplete fix rather than a genuine correction.

**Before OD (overfitting):**
```diff
@@ -12,6 +12,6 @@
 } else if (type.hasRawClass(CLASS_SINGLETON_SET)) {
 conv = converter(TYPE_SINGLETON_SET, type, Set.class);
-// [databind#2265]: we may have another impl type for unmodifiable Lists, check both
-} else if (type.hasRawClass(CLASS_UNMODIFIABLE_LIST)) {
+// Corrected handling for unmodifiable Lists
+} else if (type.isTypeOrSubTypeOf(List.class) && isUnmodifiableList(type)) {
 conv = converter(TYPE_UNMODIFIABLE_LIST, type, List.class);
 } else if (type.hasRawClass(CLASS_UNMODIFIABLE_SET)) {
@@ -22,2 +22,6 @@
 return new StdDelegatingDeserializer<Object>(conv);
 }
+
+private static boolean isUnmodifiableList(JavaType type) {
+return type.getRawClass().getName().equals("java.util.Collections$UnmodifiableList");
+}
```

**After OD (correct):**
```diff
@@ -1,9 +1,6 @@
-public static JsonDeserializer<?> findForCollection(DeserializationContext ctxt,
-JavaType type)
-throws JsonMappingException
-{
+public static JsonDeserializer<?> findForCollection(DeserializationContext ctxt, JavaType type) throws JsonMappingException {
 JavaUtilCollectionsConverter conv;
 
-// 10-Jan-2017, tatu: Some types from `java.util.Collections`/`java.util.Arrays` need bit of help...
+// Handle types from java.util.Collections/java.util.Arrays that need conversion
 if (type.hasRawClass(CLASS_AS_ARRAYS_LIST)) {
 conv = converter(TYPE_AS_LIST, type, List.class);
@@ -12,9 +9,14 @@
 } else if (type.hasRawClass(CLASS_SINGLETON_SET)) {
 conv = converter(TYPE_SINGLETON_SET, type, Set.class);
-// [databind#2265]: we may have another impl type for unmodifiable Lists, check both
-} else if (type.hasRawClass(CLASS_UNMODIFIABLE_LIST)) {
+} else if (isUnmodifiableCollection(type)) {
+if (type.isTypeOrSubTypeOf(List.class)) {
 conv = converter(TYPE_UNMODIFIABLE_LIST, type, List.class);
-} else if (type.hasRawClass(CLASS_UNMODIFIABLE_SET)) {
+} else if (type.isTypeOrSubTypeOf(Set.class)) {
 conv = converter(TYPE_UNMODIFIABLE_SET, type, Set.class);
+} else if (type.isTypeOrSubTypeOf(Map.class)) {
+conv = converter(TYPE_UNMODIFIABLE_MAP, type, Map.class);
+} else {
+return null;
+}
 } else {
 return null;
@@ -22,2 +24,7 @@
 return new StdDelegatingDeserializer<Object>(conv);
 }
+
+private static boolean isUnmodifiableCollection(JavaType type) {
+String className = type.getRawClass().getName();
+return className.startsWith("java.util.Collections$Unmodifiable");
+}
```

**Ground truth fix:**
```diff
@@ -13,5 +13,5 @@
 conv = converter(TYPE_SINGLETON_SET, type, Set.class);
 // [databind#2265]: we may have another impl type for unmodifiable Lists, check both
-} else if (type.hasRawClass(CLASS_UNMODIFIABLE_LIST)) {
+} else if (type.hasRawClass(CLASS_UNMODIFIABLE_LIST) || type.hasRawClass(CLASS_UNMODIFIABLE_LIST_ALIAS)) {
 conv = converter(TYPE_UNMODIFIABLE_LIST, type, List.class);
 } else if (type.hasRawClass(CLASS_UNMODIFIABLE_SET)) {
```

---

## JacksonDatabind-77

**Root cause:** The bug is that bean deserializer creation did not invoke the illegal-type security check before building a bean deserializer, allowing blocked gadget classes to be deserialized as beans.

**Why pre-patch overfits:** The pre-patch does not implement the actual root-cause fix in the same way as the ground-truth patch, which adds a general checkIllegalTypes(ctxt, type, beanDesc) at the bean-deserializer path. Instead, it overfits to the observed test by hardcoding com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl, so it would miss other illegal types that should also be rejected on reasonable unseen inputs.

**OD refinement (NECESSARY):** The OD refinement changed the pre-patch from blocking exactly one hardcoded class name to blocking a broader set of dangerous classes by package prefix. This is a real semantic expansion: the pre-patch only catches TemplatesImpl, whereas the refined patch attempts to address the underlying security mechanism more generally.

**Verdict:** Because the refinement introduced necessary missing generality rather than a cosmetic change, the pre-patch was a test-specific workaround. It passes the given test but does not correctly restore the intended illegal-type checking behavior.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,8 @@
 {
 final DeserializationConfig config = ctxt.getConfig();
+// Check for illegal types for security reasons
+if (isIllegalType(type)) {
+throw JsonMappingException.from(ctxt, "Illegal type \"" + type.getRawClass().getName() + "\" to deserialize; prevented for security reasons");
+}
 // We may also have custom overrides:
 JsonDeserializer<Object> custom = _findCustomBeanDeserializer(type, config, beanDesc);
@@ -49,2 +53,8 @@
 return buildBeanDeserializer(ctxt, type, beanDesc);
 }
+
+private boolean isIllegalType(JavaType type) {
+// Add checks for restricted types
+String className = type.getRawClass().getName();
+return className.startsWith("com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl");
+}
```

**After OD (correct):**
```diff
@@ -5,4 +5,8 @@
 {
 final DeserializationConfig config = ctxt.getConfig();
+// Check for illegal types in a more generalized manner
+if (isIllegalType(type)) {
+throw JsonMappingException.from(ctxt, "Illegal type '" + type.getRawClass().getName() + "' to deserialize; prevented for security reasons");
+}
 // We may also have custom overrides:
 JsonDeserializer<Object> custom = _findCustomBeanDeserializer(type, config, beanDesc);
@@ -45,6 +49,21 @@
 return null;
 }
-// For checks like [databind#1599]
 // Use generic bean introspection to build deserializer
 return buildBeanDeserializer(ctxt, type, beanDesc);
 }
+
+private boolean isIllegalType(JavaType type) {
+// Generalized check for illegal types
+String[] restrictedPackages = new String[]{
+"com.sun.org.apache.xalan.internal.xsltc.trax",
+"org.apache.commons.collections.functors"
+// More packages can be configured here
+};
+String className = type.getRawClass().getName();
+for (String pkg : restrictedPackages) {
+if (className.startsWith(pkg)) {
+return true;
+}
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -46,4 +46,5 @@
 }
 // For checks like [databind#1599]
+checkIllegalTypes(ctxt, type, beanDesc);
 // Use generic bean introspection to build deserializer
 return buildBeanDeserializer(ctxt, type, beanDesc);
```

---

## JacksonDatabind-78

**Root cause:** The bug is that bean deserializer creation omitted the illegal-type security check, allowing dangerous classes to be deserialized instead of being rejected early.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix in a general way; it adds an ad hoc `isIllegalType` that matches only one specific class name and even comments that it is just an example. It would still allow other illegal/deserialization-blocked types that should be rejected by the intended security mechanism, so it is overfitting to the observed failing test rather than providing the proper check.

**OD refinement (NECESSARY):** The OD refinement changed the illegal-type detection from a single hardcoded class check for `com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl` to a broader prefix-based blacklist covering multiple dangerous type families. This is a real semantic expansion, not cosmetic: the pre-patch only blocks the exact tested gadget class, while the refined patch attempts to address the general security problem of rejecting illegal types.

**Verdict:** Because the refinement is NECESSARY and broadens the missing security logic beyond the single tested class, the pre-patch was not already semantically correct. It passed tests by hardcoding the specific offending type instead of implementing the intended illegal-type validation.

**Before OD (overfitting):**
```diff
@@ -1,3 +1,2 @@
-@Override
 public JsonDeserializer<Object> createBeanDeserializer(DeserializationContext ctxt,
 JavaType type, BeanDescription beanDesc)
@@ -5,4 +4,8 @@
 {
 final DeserializationConfig config = ctxt.getConfig();
+// Check for illegal types to prevent security issues
+if (isIllegalType(type)) {
+throw new JsonMappingException(ctxt.getParser(), "Illegal type " + type + " to deserialize, prevented for security reasons");
+}
 // We may also have custom overrides:
 JsonDeserializer<Object> custom = _findCustomBeanDeserializer(type, config, beanDesc);
@@ -48,2 +51,8 @@
 return buildBeanDeserializer(ctxt, type, beanDesc);
 }
+
+private boolean isIllegalType(JavaType type) {
+String className = type.getRawClass().getName();
+// Example check, should be replaced with the real security restrictions
+return className.startsWith("com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl");
+}
```

**After OD (correct):**
```diff
@@ -1,3 +1,2 @@
-@Override
 public JsonDeserializer<Object> createBeanDeserializer(DeserializationContext ctxt,
 JavaType type, BeanDescription beanDesc)
@@ -5,4 +4,8 @@
 {
 final DeserializationConfig config = ctxt.getConfig();
+// Check for illegal types based on disallowed package prefixes
+if (isIllegalType(type)) {
+throw new JsonMappingException(ctxt.getParser(), "Illegal type " + type + " to deserialize, prevented for security reasons");
+}
 // We may also have custom overrides:
 JsonDeserializer<Object> custom = _findCustomBeanDeserializer(type, config, beanDesc);
@@ -48,2 +51,19 @@
 return buildBeanDeserializer(ctxt, type, beanDesc);
 }
+
+private boolean isIllegalType(JavaType type) {
+String className = type.getRawClass().getName();
+// Check for illegal types by prefix
+String[] disallowedPackages = {
+"com.sun.org.apache.xalan.internal.xsltc.trax.",
+"java.net.",
+"org.apache.commons.",
+// Add other disallowed package prefixes here
+};
+for (String prefix : disallowedPackages) {
+if (className.startsWith(prefix)) {
+return true;
+}
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -45,4 +45,5 @@
 }
 // For checks like [databind#1599]
+checkIllegalTypes(ctxt, type, beanDesc);
 // Use generic bean introspection to build deserializer
 return buildBeanDeserializer(ctxt, type, beanDesc);
```

---

## JacksonDatabind-88

**Root cause:** For canonical generic type ids, the method constructed a JavaType directly without verifying that the resolved raw class is compatible with the declared base type, allowing invalid non-subtypes through until a later mapping failure.

**Why pre-patch overfits:** The pre-patch does address the original bug by adding subtype validation for generic canonical ids, and even applies the check uniformly to both generic and non-generic paths. However, it is too strict: for reasonable inputs where `_baseType` is `Object`, `constructedType.isTypeOrSubTypeOf(Object.class)` may not behave as intended for all JavaType forms, and the confirmed-correct refinement explicitly treats Object as a special case to avoid rejecting valid ids. So the pre-patch is not a fully correct general fix.

**OD refinement (NECESSARY):** The refinement added a guard to skip the subtype check when the base type is Object.class: `!_baseType.hasRawClass(Object.class) && ...`. This is a real semantic change, because the pre-patch would reject valid type ids whenever the polymorphic base type is Object, while the refined patch preserves the intended permissive behavior for that common case. The other message-text edits are cosmetic.

**Verdict:** Because the OD refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was over-restrictive and not semantically correct in general. The confirmed-correct patch's Object-base exemption shows the pre-patch's subtype check was incomplete.

**Before OD (overfitting):**
```diff
@@ -1,34 +1,25 @@
-protected JavaType _typeFromId(String id, DatabindContext ctxt) throws IOException
-{
-/* 30-Jan-2010, tatu: Most ids are basic class names; so let's first
-*    check if any generics info is added; and only then ask factory
-*    to do translation when necessary
-*/
+protected JavaType _typeFromId(String id, DatabindContext ctxt) throws IOException {
 TypeFactory tf = ctxt.getTypeFactory();
+JavaType constructedType;
 if (id.indexOf('<') > 0) {
-// note: may want to try combining with specialization (esp for EnumMap)?
-// 17-Aug-2017, tatu: As per [databind#1735] need to ensure assignment
-//    compatibility -- needed later anyway, and not doing so may open
-//    security issues.
-JavaType t = tf.constructFromCanonical(id);
-// Probably cleaner to have a method in `TypeFactory` but can't add in patch
-return t;
-}
+constructedType = tf.constructFromCanonical(id);
+} else {
 Class<?> cls;
 try {
-cls =  tf.findClass(id);
+cls = tf.findClass(id);
 } catch (ClassNotFoundException e) {
-// 24-May-2016, tatu: Ok, this is pretty ugly, but we should always get
-//   DeserializationContext, just playing it safe
 if (ctxt instanceof DeserializationContext) {
 DeserializationContext dctxt = (DeserializationContext) ctxt;
-// First: we may have problem handlers that can deal with it?
 return dctxt.handleUnknownTypeId(_baseType, id, this, "no such class found");
 }
-// ... meaning that we really should never get here.
 return null;
 } catch (Exception e) {
 throw new IllegalArgumentException("Invalid type id '"+id+"' (for id type 'Id.class'): "+e.getMessage(), e);
 }
-return tf.constructSpecializedType(_baseType, cls);
+constructedType = tf.constructSpecializedType(_baseType, cls);
 }
+if (!constructedType.isTypeOrSubTypeOf(_baseType.getRawClass())) {
+throw new IllegalArgumentException("Type '"+id+"' is not subtype of base type '"+_baseType+"'");
+}
+return constructedType;
+}
```

**After OD (correct):**
```diff
@@ -1,34 +1,25 @@
-protected JavaType _typeFromId(String id, DatabindContext ctxt) throws IOException
-{
-/* 30-Jan-2010, tatu: Most ids are basic class names; so let's first
-*    check if any generics info is added; and only then ask factory
-*    to do translation when necessary
-*/
+protected JavaType _typeFromId(String id, DatabindContext ctxt) throws IOException {
 TypeFactory tf = ctxt.getTypeFactory();
+JavaType constructedType;
 if (id.indexOf('<') > 0) {
-// note: may want to try combining with specialization (esp for EnumMap)?
-// 17-Aug-2017, tatu: As per [databind#1735] need to ensure assignment
-//    compatibility -- needed later anyway, and not doing so may open
-//    security issues.
-JavaType t = tf.constructFromCanonical(id);
-// Probably cleaner to have a method in `TypeFactory` but can't add in patch
-return t;
-}
+constructedType = tf.constructFromCanonical(id);
+} else {
 Class<?> cls;
 try {
-cls =  tf.findClass(id);
+cls = tf.findClass(id);
 } catch (ClassNotFoundException e) {
-// 24-May-2016, tatu: Ok, this is pretty ugly, but we should always get
-//   DeserializationContext, just playing it safe
 if (ctxt instanceof DeserializationContext) {
 DeserializationContext dctxt = (DeserializationContext) ctxt;
-// First: we may have problem handlers that can deal with it?
-return dctxt.handleUnknownTypeId(_baseType, id, this, "no such class found");
+return dctxt.handleUnknownTypeId(_baseType, id, this, "No such class found: " + id);
 }
-// ... meaning that we really should never get here.
 return null;
 } catch (Exception e) {
-throw new IllegalArgumentException("Invalid type id '"+id+"' (for id type 'Id.class'): "+e.getMessage(), e);
+throw new IllegalArgumentException("Invalid type id '"+id+"' (for id type 'Id.class'): " + e.getMessage(), e);
 }
-return tf.constructSpecializedType(_baseType, cls);
+constructedType = tf.constructSpecializedType(_baseType, cls);
 }
+if (!_baseType.hasRawClass(Object.class) && !constructedType.isTypeOrSubTypeOf(_baseType.getRawClass())) {
+throw new IllegalArgumentException("Type '" + id + "' is not subtype of base type '" + _baseType + "'");
+}
+return constructedType;
+}
```

**Ground truth fix:**
```diff
@@ -12,5 +12,9 @@
 //    security issues.
 JavaType t = tf.constructFromCanonical(id);
+if (!t.isTypeOrSubTypeOf(_baseType.getRawClass())) {
 // Probably cleaner to have a method in `TypeFactory` but can't add in patch
+throw new IllegalArgumentException(String.format(
+"Class %s not subtype of %s", t.getRawClass().getName(), _baseType));
+}
 return t;
 }
```

---

## Jsoup-19

**Root cause:** The bug is that the method validates only the absolute URL form, so when absUrl() is empty for non-resolvable/custom protocols, it discards the original attribute value instead of checking that original value against the allowed protocols.

**Why pre-patch overfits:** The pre-patch partially addresses the root cause by falling back to attr.getValue() when absUrl() is empty, which is the essential fix for custom protocols like cid: and data:. However, it also adds a semantically wrong branch: for values lacking a colon, it attempts Protocol.valueOf(...) on the original attribute prefix, which can incorrectly accept/reject inputs and can throw IllegalArgumentException for ordinary relative URLs or unknown prefixes, so it is not a generally correct fix.

**OD refinement (NECESSARY):** The refinement removes the extra fallback condition that tries to accept values without a colon by parsing the original attribute prefix via Protocol.valueOf(...), and instead cleanly uses the original attribute value whenever no absolute URL is available. That change is necessary because the pre-patch introduced incorrect protocol-validation behavior and potential runtime failure paths unrelated to the real fix.

**Verdict:** Because the OD refinement made a NECESSARY semantic correction by removing the erroneous extra acceptance logic and using the original value in a safe, general way, the pre-patch was a true overfitting/incorrect patch rather than an already-correct alternative.

**Before OD (overfitting):**
```diff
@@ -1,12 +1,16 @@
 private boolean testValidProtocol(Element el, Attribute attr, Set<Protocol> protocols) {
 // try to resolve relative urls to abs, and optionally update the attribute so output html has abs.
-// rels without a baseuri get removed
+// rels without a baseuri get removed unless allowed by a protocol
 String value = el.absUrl(attr.getKey());
-if (!preserveRelativeLinks)
-attr.setValue(value);
-
+if (!preserveRelativeLinks) {
+if(value.isEmpty()) {
+value = attr.getValue(); // Use the original attribute if absUrl is empty
+} else {
+attr.setValue(value); // Set the absolute URL if obtainable
+}
+}
 for (Protocol protocol : protocols) {
 String prot = protocol.toString() + ":";
-if (value.toLowerCase().startsWith(prot)) {
+if (value.toLowerCase().startsWith(prot) || (!value.contains(":") && protocols.contains(Protocol.valueOf(attr.getValue().toLowerCase().split(":")[0])))) {
 return true;
 }
```

**After OD (correct):**
```diff
@@ -1,8 +1,13 @@
 private boolean testValidProtocol(Element el, Attribute attr, Set<Protocol> protocols) {
-// try to resolve relative urls to abs, and optionally update the attribute so output html has abs.
-// rels without a baseuri get removed
-String value = el.absUrl(attr.getKey());
-if (!preserveRelativeLinks)
-attr.setValue(value);
+// Try to resolve relative URLs to absolute, and optionally update the attribute so output HTML has abs.
+// Rels without a base URI do not have an absolute path and should remain relative.
+String absValue = el.absUrl(attr.getKey());
+String value = attr.getValue();
+
+if (!preserveRelativeLinks && !absValue.isEmpty()) {
+// Only set the value if an absolute URL can be obtained
+attr.setValue(absValue);
+value = absValue; // Use the absolute URL if obtainable
+}
 
 for (Protocol protocol : protocols) {
@@ -12,4 +17,5 @@
 }
 }
+
 return false;
 }
```

**Ground truth fix:**
```diff
@@ -3,4 +3,6 @@
 // rels without a baseuri get removed
 String value = el.absUrl(attr.getKey());
+if (value.length() == 0)
+value = attr.getValue(); // if it could not be made abs, run as-is to allow custom unknown protocols
 if (!preserveRelativeLinks)
 attr.setValue(value);
```

---

## Jsoup-50

**Root cause:** The bug is that parseByteData did not detect and honor leading BOMs before decoding, so UTF-16/UTF-32 inputs were initially decoded with the wrong charset and parsed incorrectly.

**Why pre-patch overfits:** The pre-patch does address the main failing tests for UTF-16/UTF-32 by detecting those BOMs and decoding with the corresponding charset, so it is not merely test-only. However, it is semantically incomplete because it mishandles a reasonable input class: files with a UTF-8 BOM will not be recognized, leaving the BOM character in the decoded text instead of consuming it as required. That means the pre-patch still produces incorrect results on valid BOM-encoded inputs beyond the provided tests.

**OD refinement (NECESSARY):** The refinement corrected the BOM handling logic so that absence of a BOM does not disturb normal decoding, and it also fixed UTF-8 BOM detection/reset behavior. In the pre-patch, detectCharsetFromBom checks for UTF-8 using the impossible 4-byte pattern 0xEFBBBF00, so UTF-8 BOMs are not detected, and the method also eagerly decodes immediately in the BOM branch instead of just setting charset/position as in the refined version.

**Verdict:** Because the refinement introduced necessary logic?봫ost importantly, correct UTF-8 BOM detection/reset semantics?봳he pre-patch was not a fully correct fix. It passed the given tests but remained wrong for other BOM cases, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -3,8 +3,11 @@
 Document doc = null;
 
-// look for BOM - overrides any other header or input
-
-if (charsetName == null) { // determine from meta. safe parse as UTF-8
-// look for <meta http-equiv="Content-Type" content="text/html;charset=gb2312"> or HTML5 <meta charset="gb2312">
+// Detect BOM and adjust charset accordingly
+BomCharset bomCharset = detectCharsetFromBom(byteData);
+if (bomCharset != null) {
+charsetName = bomCharset.getCharsetName();
+byteData.position(bomCharset.getBomLength()); // skip BOM
+docData = Charset.forName(charsetName).decode(byteData).toString();
+} else if (charsetName == null) { // determine from meta. safe parse as UTF-8
 docData = Charset.forName(defaultCharset).decode(byteData).toString();
 doc = parser.parseInput(docData, baseUri);
@@ -37,11 +40,5 @@
 docData = Charset.forName(charsetName).decode(byteData).toString();
 }
-if (docData.length() > 0 && docData.charAt(0) == UNICODE_BOM) {
-byteData.rewind();
-docData = Charset.forName(defaultCharset).decode(byteData).toString();
-docData = docData.substring(1);
-charsetName = defaultCharset;
-doc = null;
-}
+
 if (doc == null) {
 doc = parser.parseInput(docData, baseUri);
@@ -50,2 +47,46 @@
 return doc;
 }
+
+// Helper class to detect BOM and return charset details
+static class BomCharset {
+private final String charsetName;
+private final int bomLength;
+
+public BomCharset(String charsetName, int bomLength) {
+this.charsetName = charsetName;
+this.bomLength = bomLength;
+}
+
+public String getCharsetName() {
+return charsetName;
+}
+
+public int getBomLength() {
+return bomLength;
+}
+}
+
+// Detects the charset from BOM
+static BomCharset detectCharsetFromBom(ByteBuffer byteData) {
+if (byteData.remaining() >= 4) {
+int first4 = byteData.getInt(byteData.position());
+switch (first4) {
+case 0x0000FEFF:
+return new BomCharset("UTF-32BE", 4);
+case 0xFFFE0000:
+return new BomCharset("UTF-32LE", 4);
+case 0xEFBBBF00:
+return new BomCharset("UTF-8", 3);
+}
+}
+if (byteData.remaining() >= 2) {
+int first2 = byteData.getShort(byteData.position()) & 0xFFFF;
+switch (first2) {
+case 0xFEFF:
+return new BomCharset("UTF-16BE", 2);
+case 0xFFFE:
+return new BomCharset("UTF-16LE", 2);
+}
+}
+return null;
+}
```

**After OD (correct):**
```diff
@@ -3,12 +3,19 @@
 Document doc = null;
 
-// look for BOM - overrides any other header or input
+byteData.mark(); // mark current position to reset if necessary
+BomCharset bomCharset = detectCharsetFromBom(byteData);
+if (bomCharset != null) {
+charsetName = bomCharset.getCharsetName();
+byteData.position(bomCharset.getBomLength()); // skip BOM
+} else {
+byteData.reset();
+}
 
-if (charsetName == null) { // determine from meta. safe parse as UTF-8
-// look for <meta http-equiv="Content-Type" content="text/html;charset=gb2312"> or HTML5 <meta charset="gb2312">
-docData = Charset.forName(defaultCharset).decode(byteData).toString();
+if (charsetName == null) {
+charsetName = defaultCharset;
+docData = Charset.forName(charsetName).decode(byteData).toString();
 doc = parser.parseInput(docData, baseUri);
 Element meta = doc.select("meta[http-equiv=content-type], meta[charset]").first();
-if (meta != null) { // if not found, will keep utf-8 as best attempt
+if (meta != null) {
 String foundCharset = null;
 if (meta.hasAttr("http-equiv")) {
@@ -25,5 +32,5 @@
 }
 
-if (foundCharset != null && foundCharset.length() != 0 && !foundCharset.equals(defaultCharset)) { // need to re-decode
+if (foundCharset != null && !foundCharset.equals(defaultCharset)) {
 foundCharset = foundCharset.trim().replaceAll("[\"']", "");
 charsetName = foundCharset;
@@ -33,15 +40,9 @@
 }
 }
-} else { // specified by content type header (or by user on file load)
+} else {
 Validate.notEmpty(charsetName, "Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML");
 docData = Charset.forName(charsetName).decode(byteData).toString();
 }
-if (docData.length() > 0 && docData.charAt(0) == UNICODE_BOM) {
-byteData.rewind();
-docData = Charset.forName(defaultCharset).decode(byteData).toString();
-docData = docData.substring(1);
-charsetName = defaultCharset;
-doc = null;
-}
+
 if (doc == null) {
 doc = parser.parseInput(docData, baseUri);
@@ -50,2 +51,46 @@
 return doc;
 }
+
+static BomCharset detectCharsetFromBom(ByteBuffer byteData) {
+byteData.mark(); // mark current position
+if (byteData.remaining() >= 4) {
+int first4 = byteData.getInt(byteData.position());
+switch (first4) {
+case 0x0000FEFF:
+return new BomCharset("UTF-32BE", 4);
+case 0xFFFE0000:
+return new BomCharset("UTF-32LE", 4);
+case 0xEFBBBF:
+return new BomCharset("UTF-8", 3);
+}
+}
+if (byteData.remaining() >= 2) {
+int first2 = byteData.getShort(byteData.position()) & 0xFFFF;
+switch (first2) {
+case 0xFEFF:
+return new BomCharset("UTF-16BE", 2);
+case 0xFFFE:
+return new BomCharset("UTF-16LE", 2);
+}
+}
+byteData.reset(); // reset to original position if no BOM is detected
+return null;
+}
+
+static class BomCharset {
+private final String charsetName;
+private final int bomLength;
+
+public BomCharset(String charsetName, int bomLength) {
+this.charsetName = charsetName;
+this.bomLength = bomLength;
+}
+
+public String getCharsetName() {
+return charsetName;
+}
+
+public int getBomLength() {
+return bomLength;
+}
+}
```

**Ground truth fix:**
```diff
@@ -4,4 +4,18 @@
 
 // look for BOM - overrides any other header or input
+byteData.mark();
+byte[] bom = new byte[4];
+byteData.get(bom);
+byteData.rewind();
+if (bom[0] == 0x00 && bom[1] == 0x00 && bom[2] == (byte) 0xFE && bom[3] == (byte) 0xFF || // BE
+bom[0] == (byte) 0xFF && bom[1] == (byte) 0xFE && bom[2] == 0x00 && bom[3] == 0x00) { // LE
+charsetName = "UTF-32"; // and I hope it's on your system
+} else if (bom[0] == (byte) 0xFE && bom[1] == (byte) 0xFF || // BE
+bom[0] == (byte) 0xFF && bom[1] == (byte) 0xFE) {
+charsetName = "UTF-16"; // in all Javas
+} else if (bom[0] == (byte) 0xEF && bom[1] == (byte) 0xBB && bom[2] == (byte) 0xBF) {
+charsetName = "UTF-8"; // in all Javas
+byteData.position(3); // 16 and 32 decoders consume the BOM to determine be/le; utf-8 should be consumed
+}
 
 if (charsetName == null) { // determine from meta. safe parse as UTF-8
@@ -37,11 +51,4 @@
 docData = Charset.forName(charsetName).decode(byteData).toString();
 }
-if (docData.length() > 0 && docData.charAt(0) == UNICODE_BOM) {
-byteData.rewind();
-docData = Charset.forName(defaultCharset).decode(byteData).toString();
-docData = docData.substring(1);
-charsetName = defaultCharset;
-doc = null;
-}
 if (doc == null) {
 doc = parser.parseInput(docData, baseUri);
```

---

## Jsoup-53

**Root cause:** chompBalanced incorrectly counts open/close delimiter characters that appear inside quoted attribute values, so selectors like [data='End]'] are parsed as prematurely closed.

**Why pre-patch overfits:** The pre-patch fixes the reported tests by ignoring brackets while inside quotes, but it does so incompletely: inside a single-quoted region, an unescaped double quote toggles inQuote off (and vice versa). On reasonable inputs such as mixed-quote attribute values containing the opposite quote plus a closing bracket later, it can incorrectly decrement depth and terminate early, so it does not robustly fix the root cause.

**OD refinement (NECESSARY):** The refinement changed quote handling from a single boolean toggle on any unescaped quote to tracking the specific active quote character and only ending quote mode when the matching quote is seen. This is semantically important because the pre-patch treats a different quote character inside a quoted string as closing the quote, which can re-enable bracket counting too early and misparse balanced content.

**Verdict:** Because the refinement introduced necessary logic rather than a cosmetic refactor, and that logic addresses a real missing semantic condition in the pre-patch, the original patch was an overfitting/incomplete fix rather than a genuinely correct one.

**Before OD (overfitting):**
```diff
@@ -4,20 +4,31 @@
 int depth = 0;
 char last = 0;
+boolean inQuote = false;
 
 do {
 if (isEmpty()) break;
 Character c = consume();
+if (c == '"' || c == '\'') {
+if (last != ESC) {
+inQuote = !inQuote;
+}
+}
+if (!inQuote) {
 if (last == 0 || last != ESC) {
 if (c.equals(open)) {
 depth++;
-if (start == -1)
+if (start == -1) {
 start = pos;
 }
-else if (c.equals(close))
+} else if (c.equals(close)) {
 depth--;
 }
+}
 
-if (depth > 0 && last != 0)
+if (depth > 0 && last != 0) {
 end = pos; // don't include the outer match pair in the return
+}
+}
+
 last = c;
 } while (depth > 0);
```

**After OD (correct):**
```diff
@@ -4,20 +4,38 @@
 int depth = 0;
 char last = 0;
+boolean inQuote = false;
+char quoteChar = 0;
 
 do {
 if (isEmpty()) break;
 Character c = consume();
+
+// Quote handling
+if ((c == '"' || c == '\'') && (last != ESC || last == 0)) {
+if (!inQuote) {
+inQuote = true;
+quoteChar = c;
+} else if (c.equals(quoteChar)) {
+inQuote = false;
+}
+}
+
+if (!inQuote) {
 if (last == 0 || last != ESC) {
 if (c.equals(open)) {
 depth++;
-if (start == -1)
+if (start == -1) {
 start = pos;
 }
-else if (c.equals(close))
+} else if (c.equals(close)) {
 depth--;
 }
+}
 
-if (depth > 0 && last != 0)
+if (depth > 0 && last != 0) {
 end = pos; // don't include the outer match pair in the return
+}
+}
+
 last = c;
 } while (depth > 0);
```

**Ground truth fix:**
```diff
@@ -4,4 +4,5 @@
 int depth = 0;
 char last = 0;
+boolean inQuote = false;
 
 do {
@@ -9,4 +10,8 @@
 Character c = consume();
 if (last == 0 || last != ESC) {
+if (c.equals('\'') || c.equals('"') && c != open)
+inQuote = !inQuote;
+if (inQuote)
+continue;
 if (c.equals(open)) {
 depth++;
```

---

## Lang-19

**Root cause:** The bug is that numeric entity parsing assumes a semicolon-terminated entity and reads past the end or through non-digit characters instead of safely stopping at string bounds and valid numeric characters.

**Why pre-patch overfits:** The pre-patch addresses some boundary cases and supports missing semicolons, but it does not correctly fix the root cause because its scanning loop is unsafe for hexadecimal entities. For inputs like an unfinished hex entity at the end of the string (e.g. `"&#x30"` or similar cases), once `end` reaches `seqEnd`, the right side of the `||` still evaluates and calls `input.charAt(end)`, so the patch can still throw `StringIndexOutOfBoundsException` outside the provided tests.

**OD refinement (NECESSARY):** The refinement fixes the pre-patch's malformed loop condition by properly grouping the bounds check with the digit tests: `end < seqEnd && (...)`. In the pre-patch, operator precedence makes the condition effectively `(end < seqEnd && Character.isDigit(...)) || (isHex && Character.digit(...) >= 0)`, so for hex entities it can evaluate `input.charAt(end)` even when `end == seqEnd`, causing out-of-bounds access and incorrect behavior on reasonable inputs.

**Verdict:** Because the OD refinement introduced necessary logic to make the bounds check semantically correct, the pre-patch was not equivalent and remained wrong on plausible inputs beyond the tests. Therefore the original patch was a true overfitting/incorrect fix.

**Before OD (overfitting):**
```diff
@@ -6,16 +6,19 @@
 boolean isHex = false;
 
+if (start >= seqEnd) {
+return 0;
+}
+
 char firstChar = input.charAt(start);
 if(firstChar == 'x' || firstChar == 'X') {
 start++;
 isHex = true;
-
-// Check there's more than just an x after the &#
+if (start >= seqEnd) {
+return 0;
+}
 }
 
 int end = start;
-// Note that this supports character codes without a ; on the end
-while(input.charAt(end) != ';')
-{
+while(end < seqEnd && Character.isDigit(input.charAt(end)) || (isHex && Character.digit(input.charAt(end), 16) >= 0)) {
 end++;
 }
@@ -23,4 +26,5 @@
 int entityValue;
 try {
+if(end > start) { // Ensure there's at least one digit
 if(isHex) {
 entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 16);
@@ -28,6 +32,8 @@
 entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 10);
 }
+} else {
+return 0;
+}
 } catch(NumberFormatException nfe) {
-System.err.println("FAIL: " + input.subSequence(start, end) + "[" + start +"]["+ end +"]");
 return 0;
 }
@@ -41,6 +47,10 @@
 }
 
+// Account ';' if it exists
+if (end < seqEnd && input.charAt(end) == ';') {
+end++;
+}
 
-return 2 + (end - start) + (isHex ? 1 : 0) + 1;
+return end - index;
 }
 return 0;
```

**After OD (correct):**
```diff
@@ -1,37 +1,43 @@
 public int translate(CharSequence input, int index, Writer out) throws IOException {
 int seqEnd = input.length();
-// Uses -2 to ensure there is something after the &#
-if(input.charAt(index) == '&' && index < seqEnd - 1 && input.charAt(index + 1) == '#') {
+if (index < seqEnd - 2 && input.charAt(index) == '&' && input.charAt(index + 1) == '#') {
 int start = index + 2;
 boolean isHex = false;
 
+if (start >= seqEnd) {
+return 0;
+}
+
 char firstChar = input.charAt(start);
-if(firstChar == 'x' || firstChar == 'X') {
+if (firstChar == 'x' || firstChar == 'X') {
 start++;
 isHex = true;
-
-// Check there's more than just an x after the &#
+if (start >= seqEnd) {
+return 0;
+}
 }
 
 int end = start;
-// Note that this supports character codes without a ; on the end
-while(input.charAt(end) != ';')
-{
+while (end < seqEnd && ((isHex && Character.digit(input.charAt(end), 16) >= 0) || (!isHex && Character.isDigit(input.charAt(end))))) {
 end++;
+}
+
+if (end == start) { // No valid digits were found
+return 0;
 }
 
 int entityValue;
 try {
-if(isHex) {
-entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 16);
+String numberSequence = input.subSequence(start, end).toString();
+if (isHex) {
+entityValue = Integer.parseInt(numberSequence, 16);
 } else {
-entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 10);
+entityValue = Integer.parseInt(numberSequence, 10);
 }
-} catch(NumberFormatException nfe) {
-System.err.println("FAIL: " + input.subSequence(start, end) + "[" + start +"]["+ end +"]");
+} catch (NumberFormatException nfe) {
 return 0;
 }
 
-if(entityValue > 0xFFFF) {
+if (entityValue > 0xFFFF) {
 char[] chrs = Character.toChars(entityValue);
 out.write(chrs[0]);
@@ -41,6 +47,9 @@
 }
 
+if (end < seqEnd && input.charAt(end) == ';') {
+end++;
+}
 
-return 2 + (end - start) + (isHex ? 1 : 0) + 1;
+return end - index;
 }
 return 0;
```

**Ground truth fix:**
```diff
@@ -2,5 +2,5 @@
 int seqEnd = input.length();
 // Uses -2 to ensure there is something after the &#
-if(input.charAt(index) == '&' && index < seqEnd - 1 && input.charAt(index + 1) == '#') {
+if(input.charAt(index) == '&' && index < seqEnd - 2 && input.charAt(index + 1) == '#') {
 int start = index + 2;
 boolean isHex = false;
@@ -12,9 +12,14 @@
 
 // Check there's more than just an x after the &#
+if(start == seqEnd) {
+return 0;
+}
 }
 
 int end = start;
 // Note that this supports character codes without a ; on the end
-while(input.charAt(end) != ';')
+while(end < seqEnd && ( (input.charAt(end) >= '0' && input.charAt(end) <= '9') ||
+(input.charAt(end) >= 'a' && input.charAt(end) <= 'f') ||
+(input.charAt(end) >= 'A' && input.charAt(end) <= 'F') ) )
 {
 end++;
@@ -41,6 +46,7 @@
 }
 
+boolean semiNext = (end != seqEnd) && (input.charAt(end) == ';');
 
-return 2 + (end - start) + (isHex ? 1 : 0) + 1;
+return 2 + (end - start) + (isHex ? 1 : 0) + (semiNext ? 1 : 0);
 }
 return 0;
```

---

## Lang-38

**Root cause:** When a forced time zone is used, cloning a Calendar without first materializing its current fields/time can leave it inconsistent, so simply changing the clone's time zone formats the original wall-clock fields instead of the same instant in the target zone.

**Why pre-patch overfits:** The pre-patch is not a correct alternative fix for the root cause. It adjusts the instant by `targetOffset - sourceOffset` but leaves the cloned calendar in the original time zone, so it only simulates the desired display for some cases and relies on offset arithmetic rather than actual Calendar time-zone conversion; this can produce incorrect results for reasonable inputs involving DST transitions or other zone-rule nuances.

**OD refinement (NECESSARY):** The OD refinement replaced a manual millisecond adjustment based on raw zone offsets with the correct semantic operation: keep the same instant (`setTimeInMillis(calendar.getTimeInMillis())`) while changing the clone's time zone. This is necessary because the pre-patch never sets the clone's time zone to `mTimeZone` and instead bakes in an offset difference, which is not equivalent to Calendar's own zone conversion semantics and can be wrong around DST/zone-rule boundaries.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was a true overfitting patch. It passed the observed tests by coincidence through offset manipulation, not by correctly formatting the same instant in the forced time zone.

**Before OD (overfitting):**
```diff
@@ -1,6 +1,7 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
 if (mTimeZoneForced) {
-calendar = (Calendar) calendar.clone();
-calendar.setTimeZone(mTimeZone);
+Calendar clonedCalendar = (Calendar) calendar.clone();
+clonedCalendar.setTimeInMillis(calendar.getTimeInMillis() + mTimeZone.getOffset(calendar.getTimeInMillis()) - calendar.getTimeZone().getOffset(calendar.getTimeInMillis()));
+return applyRules(clonedCalendar, buf);
 }
 return applyRules(calendar, buf);
```

**After OD (correct):**
```diff
@@ -1,6 +1,8 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
 if (mTimeZoneForced) {
-calendar = (Calendar) calendar.clone();
-calendar.setTimeZone(mTimeZone);
+Calendar clonedCalendar = (Calendar) calendar.clone();
+clonedCalendar.setTimeZone(mTimeZone);
+clonedCalendar.setTimeInMillis(calendar.getTimeInMillis());
+return applyRules(clonedCalendar, buf);
 }
 return applyRules(calendar, buf);
```

**Ground truth fix:**
```diff
@@ -1,4 +1,5 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
 if (mTimeZoneForced) {
+calendar.getTime(); /// LANG-538
 calendar = (Calendar) calendar.clone();
 calendar.setTimeZone(mTimeZone);
```

---

