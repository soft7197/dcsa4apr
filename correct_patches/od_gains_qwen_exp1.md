# Overfitting Detector Gains - QwenCoder Exp 1

Overfitting detector gain bugs: 29

These bugs had **no correct patches before OD refinement**. All pre-refinement patches were overfitting ??they passed tests but did not actually fix the root cause. OD refinement produced correct patches for these bugs.

---

## Chart-9

**Root cause:** The bug occurs when the requested end period is not present and falls before the computed start index, because the method only treated endIndex < 0 as empty and failed to detect the broader empty-range case endIndex < startIndex.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause; it changes the semantics of end-index computation and can mis-handle cases where `end` exactly matches an existing item by shifting the end index left by one. It also still misses the general empty-range condition when `endIndex >= 0` but `endIndex < startIndex`, so it would fail on reasonable inputs beyond the triggering test.

**OD refinement (NECESSARY):** The refinement removed the pre-patch's unconditional `getIndex(end) - 1` adjustment and restored the correct meaning of `getIndex(end)`: only convert negative insertion-point results into the previous valid index. It also added the essential empty-range check `endIndex < startIndex`, which is the real missing logic needed to avoid calling `createCopy(startIndex, endIndex)` with an invalid range.

**Verdict:** Because the OD refinement introduced necessary logic rather than a cosmetic or equivalent rewrite, the pre-patch was overfitting. Its altered end-index calculation is semantically wrong and it lacks the required `endIndex < startIndex` handling.

**Before OD (overfitting):**
```diff
@@ -20,6 +20,6 @@
 }
 }
-int endIndex = getIndex(end);
-if (endIndex < 0) {             // end period is not in original series
+int endIndex = getIndex(end) - 1;
+if (endIndex < 0 && startIndex != endIndex) {             // end period is not in original series
 endIndex = -(endIndex + 1); // this is first item AFTER end period
 endIndex = endIndex - 1;    // so this is last item BEFORE end
```

**After OD (correct):**
```diff
@@ -22,9 +22,8 @@
 int endIndex = getIndex(end);
 if (endIndex < 0) {             // end period is not in original series
-endIndex = -(endIndex + 1); // this is first item AFTER end period
-endIndex = endIndex - 1;    // so this is last item BEFORE end
+endIndex = -(endIndex + 1) - 1; // this is last item BEFORE end
+if (endIndex < startIndex) {
+emptyRange = true;
 }
-if (endIndex < 0) {
-emptyRange = true;
 }
 if (emptyRange) {
```

**Ground truth fix:**
```diff
@@ -25,5 +25,5 @@
 endIndex = endIndex - 1;    // so this is last item BEFORE end
 }
-if (endIndex < 0) {
+if ((endIndex < 0)  || (endIndex < startIndex)) {
 emptyRange = true;
 }
```

---

## Cli-12

### Patch 1

**Root cause:** The parser failed to recognize GNU-style options with attached values using '=' and split them correctly into option and value tokens, while preserving special handling for property-style options like -Dproperty=value.

**Why pre-patch overfits:** The pre-patch does fix the specific failing tests for '--foo=bar', '-foo=bar', and '-f=bar', but it does not correctly preserve the original special-option behavior or general short-option splitting logic. In particular, for a valid short option like '-Dproperty=value', it first sees '=', then treats 'Dproperty' as the option name, fails to recognize it, and falls back to treating the whole token as non-option instead of splitting into '-D' and 'property=value'.

**OD refinement (NECESSARY):** The refinement changed behavior in two important ways: it handles the '=' case before the plain hasOption(opt) check, and it fixes short-option prefix splitting from arg.substring(0, 2) to the actual first option character. This is necessary because the pre-patch still mishandles clustered short options such as '-Dkey=value' or '-fbar' by checking/splitting the wrong prefix, so it does not fully implement the intended GNU parsing behavior.

**Verdict:** Because the refinement introduced necessary parsing logic that the pre-patch lacked, the pre-patch was overfitting to the observed '=' test cases rather than fully fixing the parser. The confirmed-correct refined patch repairs behavior on reasonable additional inputs that the pre-patch would still parse incorrectly.

**Before OD (overfitting):**
```diff
@@ -28,10 +28,23 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+if (opt.indexOf('=') > 0)
 {
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+String optionPart = opt.substring(0, opt.indexOf('='));
+String valuePart = opt.substring(opt.indexOf('=') + 1);
+if (options.hasOption(optionPart))
+{
+tokens.add("-" + optionPart);
+tokens.add(valuePart);
+}
+else
+{
+eatTheRest = stopAtNonOption;
+tokens.add(arg);
+}
+}
+else if (options.hasOption(arg.substring(0, 2)))
+{
+tokens.add(arg.substring(0, 2));
+tokens.add(arg.substring(2));
 }
 else
```

**After OD (correct):**
```diff
@@ -1,5 +1,5 @@
 protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption)
 {
-List tokens = new ArrayList();
+List<String> tokens = new ArrayList<>();
 
 boolean eatTheRest = false;
@@ -21,17 +21,32 @@
 {
 String opt = Util.stripLeadingHyphens(arg);
+int eqIndex = opt.indexOf('=');
 
+if (eqIndex != -1)
+{
+String optionName = opt.substring(0, eqIndex);
+String optionValue = opt.substring(eqIndex + 1);
+
+if (options.hasOption(optionName) || (optionName.length() == 1 && options.hasOption("-" + optionName)))
+{
+tokens.add("-" + optionName);
+tokens.add(optionValue);
+}
+else
+{
+eatTheRest = stopAtNonOption;
+tokens.add(arg);
+}
+}
+else
+{
 if (options.hasOption(opt))
 {
 tokens.add(arg);
 }
-else
+else if (opt.length() > 1 && options.hasOption(opt.substring(0, 1)))
 {
-if (options.hasOption(arg.substring(0, 2)))
-{
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+tokens.add("-" + opt.substring(0, 1));
+tokens.add(opt.substring(1));
 }
 else
@@ -56,4 +71,4 @@
 }
 
-return (String[]) tokens.toArray(new String[tokens.size()]);
+return tokens.toArray(new String[0]);
 }
```

**Ground truth fix:**
```diff
@@ -28,7 +28,12 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+if (opt.indexOf('=') != -1 && options.hasOption(opt.substring(0, opt.indexOf('='))))
 {
 // the format is --foo=value or -foo=value
+tokens.add(arg.substring(0, arg.indexOf('='))); // --foo
+tokens.add(arg.substring(arg.indexOf('=') + 1)); // value
+}
+else if (options.hasOption(arg.substring(0, 2)))
+{
 // the format is a special properties option (-Dproperty=value)
 tokens.add(arg.substring(0, 2)); // -D
```

### Patch 2

**Root cause:** The bug is that GnuParser.flatten does not correctly detect and split option tokens containing '=' into the option name and value, especially for long options like '--foo=bar' and short options like '-f=bar', while preserving special short-property forms like '-Dproperty=value'.

**Why pre-patch overfits:** The pre-patch fixes the three failing tests, but it does not fully fix the root cause. It will incorrectly transform any unknown '-x=y' or '--name=value' token into a recognized-looking option/value pair even when the option does not exist, and it also mishandles special property options like '-Dkey=value' by turning them into '-Dkey' and 'value' instead of '-D' and 'key=value'.

**OD refinement (NECESSARY):** The refinement added validation that an '=' token is only split when the part before '=' is actually a recognized option, and it also restored correct handling of compact short-option forms by checking the first short option character rather than blindly splitting any token with '='. This is necessary because the pre-patch unconditionally rewrites any hyphenated argument containing '=' into '-'+name and value, which changes semantics for unrecognized options and breaks special cases such as '-Dproperty=value'.

**Verdict:** Because the refinement change is NECESSARY and adds missing semantic checks that the pre-patch lacks, the original patch was overfitting to the observed tests rather than implementing the correct parsing behavior.

**Before OD (overfitting):**
```diff
@@ -28,5 +28,10 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+if (opt.contains("=")) {
+int eqIndex = opt.indexOf('=');
+tokens.add("-" + opt.substring(0, eqIndex));
+tokens.add(opt.substring(eqIndex + 1));
+}
+else if (options.hasOption(arg.substring(0, 2)))
 {
 // the format is --foo=value or -foo=value
```

**After OD (correct):**
```diff
@@ -1,5 +1,5 @@
 protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption)
 {
-List tokens = new ArrayList();
+List<String> tokens = new ArrayList<>();
 
 boolean eatTheRest = false;
@@ -21,17 +21,32 @@
 {
 String opt = Util.stripLeadingHyphens(arg);
+int eqIndex = opt.indexOf('=');
 
+if (eqIndex != -1)
+{
+String optionName = opt.substring(0, eqIndex);
+String optionValue = opt.substring(eqIndex + 1);
+
+if (options.hasOption(optionName) || (optionName.length() == 1 && options.hasOption("-" + optionName)))
+{
+tokens.add("-" + optionName);
+tokens.add(optionValue);
+}
+else
+{
+eatTheRest = stopAtNonOption;
+tokens.add(arg);
+}
+}
+else
+{
 if (options.hasOption(opt))
 {
 tokens.add(arg);
 }
-else
+else if (opt.length() > 1 && options.hasOption(opt.substring(0, 1)))
 {
-if (options.hasOption(arg.substring(0, 2)))
-{
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+tokens.add("-" + opt.substring(0, 1));
+tokens.add(opt.substring(1));
 }
 else
@@ -56,4 +71,4 @@
 }
 
-return (String[]) tokens.toArray(new String[tokens.size()]);
+return tokens.toArray(new String[0]);
 }
```

**Ground truth fix:**
```diff
@@ -28,7 +28,12 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+if (opt.indexOf('=') != -1 && options.hasOption(opt.substring(0, opt.indexOf('='))))
 {
 // the format is --foo=value or -foo=value
+tokens.add(arg.substring(0, arg.indexOf('='))); // --foo
+tokens.add(arg.substring(arg.indexOf('=') + 1)); // value
+}
+else if (options.hasOption(arg.substring(0, 2)))
+{
 // the format is a special properties option (-Dproperty=value)
 tokens.add(arg.substring(0, 2)); // -D
```

---

## Cli-23

### Patch 1

**Root cause:** The bug is caused by wrapped-line processing reusing the previous wrap position after adding indentation, so when the remaining text cannot be wrapped past the padding boundary the loop makes no progress and can hang.

**Why pre-patch overfits:** The pre-patch avoids the specific infinite-loop scenario in the tests, but it does not correctly model indented continuation lines because wrap positions are computed on text without the indentation that will actually be emitted. This can mis-wrap lines or trigger the lastPos exception on other reasonable inputs where padding materially affects the available width, so it does not fully fix the root cause.

**OD refinement (NECESSARY):** The refinement restores the correct wrapping semantics by calling findWrapPos with start position 0 for the first line and with the padding length for subsequent indented lines, while also rebuilding each subsequent line as padding + trimmed remainder before computing the next wrap point. The pre-patch instead passes nextLineTabStop as the start position and computes wrap positions on the unpadded remainder, then adds padding only after appending, which changes where breaks are found and can produce wrong wrapping behavior.

**Verdict:** Because the OD refinement made a NECESSARY semantic correction to how wrap positions are computed relative to indentation, the pre-patch was not already correct. It passed the tests with a different, semantically wrong wrapping strategy, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -2,5 +2,5 @@
 int nextLineTabStop, String text)
 {
-int pos = findWrapPos(text, width, 0);
+int pos = findWrapPos(text, width, nextLineTabStop);
 
 if (pos == -1)
@@ -19,6 +19,6 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+text = text.substring(pos).trim();
+pos = findWrapPos(text, width, nextLineTabStop);
 
 if (pos == -1)
@@ -34,4 +34,5 @@
 
 sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+text = padding + text.substring(pos);
 }
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,8 @@
-protected StringBuffer renderWrappedText(StringBuffer sb, int width,
-int nextLineTabStop, String text)
+protected StringBuffer renderWrappedText(StringBuffer sb, int width, int nextLineTabStop, String text)
 {
+if (text == null || text.isEmpty()) {
+return sb;
+}
+
 int pos = findWrapPos(text, width, 0);
 
@@ -20,5 +23,5 @@
 int lastPos = pos;
 text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+pos = findWrapPos(text, width, padding.length());
 
 if (pos == -1)
```

**Ground truth fix:**
```diff
@@ -18,5 +18,4 @@
 while (true)
 {
-int lastPos = pos;
 text = padding + text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
@@ -27,8 +26,10 @@
 
 return sb;
-} else
-if (pos == lastPos)
-{
-throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
+}
+
+if ( (text.length() > width) && (pos == nextLineTabStop - 1) ) {
+sb.append(text);
+
+return sb;
 }
 
```

### Patch 2

**Root cause:** The bug occurs because wrapped continuation lines are rebuilt with padding included in the text being re-wrapped, so for long words or heavily indented continuations the wrap position can stop advancing and the loop can become non-terminating.

**Why pre-patch overfits:** The pre-patch avoids the infinite loop for the reported tests by changing the wrapping regime, but it is semantically wrong because continuation lines are no longer prefixed with the required padding despite the method contract/comment stating they must be. It would therefore produce incorrectly formatted help text on many wrapped descriptions, so it does not correctly fix the root cause through an alternative equivalent implementation.

**OD refinement (NECESSARY):** The refinement restores correct continuation-line semantics: it keeps the remaining text untrimmed/unmodified for wrapping, computes wrap positions from column 0 on that remaining text, and adds the indentation padding only when appending output. In contrast, the pre-patch removes the padding from output entirely and changes wrapping to start at nextLineTabStop, which alters formatting behavior and does not implement the intended fix strategy.

**Verdict:** Because the OD refinement made a NECESSARY semantic correction?봱eintroducing proper indentation and wrapping behavior rather than merely restructuring code?봳he pre-patch was an overfitting fix that passed tests while breaking expected output semantics.

**Before OD (overfitting):**
```diff
@@ -19,6 +19,6 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+text = text.substring(pos).trim();
+pos = findWrapPos(text, width, nextLineTabStop);
 
 if (pos == -1)
```

**After OD (correct):**
```diff
@@ -2,4 +2,8 @@
 int nextLineTabStop, String text)
 {
+if (text == null) {
+return sb;
+}
+
 int pos = findWrapPos(text, width, 0);
 
@@ -19,10 +23,10 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
+text = text.substring(pos);
 pos = findWrapPos(text, width, 0);
 
 if (pos == -1)
 {
-sb.append(text);
+sb.append(padding).append(rtrim(text));
 
 return sb;
@@ -33,5 +37,5 @@
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(padding).append(rtrim(text.substring(0, pos))).append(defaultNewLine);
 }
 }
```

**Ground truth fix:**
```diff
@@ -18,5 +18,4 @@
 while (true)
 {
-int lastPos = pos;
 text = padding + text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
@@ -27,8 +26,10 @@
 
 return sb;
-} else
-if (pos == lastPos)
-{
-throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
+}
+
+if ( (text.length() > width) && (pos == nextLineTabStop - 1) ) {
+sb.append(text);
+
+return sb;
 }
 
```

### Patch 3

**Root cause:** The infinite loop occurs because each wrapped continuation line is prefixed with padding before recomputing the wrap position, so for long indented text the wrap point can stay stuck at the indentation boundary and never make progress.

**Why pre-patch overfits:** The pre-patch does address the main looping bug by no longer feeding padded text back into `findWrapPos`, so it fixes the reported hang/crash for ordinary help text. However, it is not fully correct semantically because `trim()` removes leading whitespace from the remaining text on every wrapped line, so inputs with intentional leading spaces in continuation text will be reformatted incorrectly; the refined patch preserves that content.

**OD refinement (NECESSARY):** The refinement changes continuation handling from `text.substring(pos).trim()` to `text.substring(pos)` and appends `rtrim(text)` only at output time, preserving leading spaces in the remaining text instead of stripping them. It also adds a null guard, but the semantically important change is removing `trim()`, because trimming alters the content and wrapping behavior of continuation lines and can misformat inputs with meaningful leading whitespace.

**Verdict:** Because the refinement introduced a necessary semantic correction rather than a cosmetic change, the pre-patch was still overfitting: it avoided the infinite loop but did so with content-altering wrapping logic. Therefore the original pre-refinement patch was not genuinely correct.

**Before OD (overfitting):**
```diff
@@ -19,6 +19,7 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+text = text.substring(pos).trim();
+pos = findWrapPos(text, width - nextLineTabStop, 0);
+text = padding + text.substring(0, pos);
 
 if (pos == -1)
@@ -33,5 +34,5 @@
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(rtrim(text)).append(defaultNewLine);
 }
 }
```

**After OD (correct):**
```diff
@@ -2,4 +2,8 @@
 int nextLineTabStop, String text)
 {
+if (text == null) {
+return sb;
+}
+
 int pos = findWrapPos(text, width, 0);
 
@@ -19,10 +23,10 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
+text = text.substring(pos);
 pos = findWrapPos(text, width, 0);
 
 if (pos == -1)
 {
-sb.append(text);
+sb.append(padding).append(rtrim(text));
 
 return sb;
@@ -33,5 +37,5 @@
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(padding).append(rtrim(text.substring(0, pos))).append(defaultNewLine);
 }
 }
```

**Ground truth fix:**
```diff
@@ -18,5 +18,4 @@
 while (true)
 {
-int lastPos = pos;
 text = padding + text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
@@ -27,8 +26,10 @@
 
 return sb;
-} else
-if (pos == lastPos)
-{
-throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
+}
+
+if ( (text.length() > width) && (pos == nextLineTabStop - 1) ) {
+sb.append(text);
+
+return sb;
 }
 
```

### Patch 4

**Root cause:** The bug occurs when a wrapped continuation line begins with indentation and contains no breakable space before the width limit, causing findWrapPos to return the indentation boundary repeatedly and the loop to make no progress.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause; it tries to avoid the repeated position by advancing the search start into the padded text, but that changes the semantics of wrapping and relies on stale coordinates from the previous iteration. On reasonable long descriptions requiring multiple wrapped continuation lines, it can skip valid wrap opportunities, wrap at wrong positions, or still throw the RuntimeException instead of printing help correctly.

**OD refinement (NECESSARY):** The refinement changes the wrapping logic to search for the next wrap position in the unpadded remaining text and only prepend padding when appending output, instead of searching inside the already padded string from a shifted start index. This is necessary because the pre-patch's use of findWrapPos(text, width, pos + padding.length()) mixes positions from the old and new strings and can produce incorrect wrap points or trigger the same non-progress condition on other long inputs.

**Verdict:** Because the refinement introduced necessary logic rather than a cosmetic rewrite, the pre-patch was not already semantically correct. It passed the tests by using an ad hoc position shift, but it did not robustly implement the proper continuation-line wrapping behavior.

**Before OD (overfitting):**
```diff
@@ -20,5 +20,5 @@
 int lastPos = pos;
 text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+pos = findWrapPos(text, width, pos + padding.length());
 
 if (pos == -1)
```

**After OD (correct):**
```diff
@@ -2,4 +2,8 @@
 int nextLineTabStop, String text)
 {
+if (text == null || text.isEmpty()) {
+return sb;
+}
+
 int pos = findWrapPos(text, width, 0);
 
@@ -16,22 +20,23 @@
 final String padding = createPadding(nextLineTabStop);
 
+int offset = 0;
 while (true)
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
+offset += pos;
+text = text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
 
 if (pos == -1)
 {
-sb.append(text);
+sb.append(padding).append(text);
 
 return sb;
-} else
-if (pos == lastPos)
+} else if (pos == lastPos)
 {
 throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(padding).append(rtrim(text.substring(0, pos))).append(defaultNewLine);
 }
 }
```

**Ground truth fix:**
```diff
@@ -18,5 +18,4 @@
 while (true)
 {
-int lastPos = pos;
 text = padding + text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
@@ -27,8 +26,10 @@
 
 return sb;
-} else
-if (pos == lastPos)
-{
-throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
+}
+
+if ( (text.length() > width) && (pos == nextLineTabStop - 1) ) {
+sb.append(text);
+
+return sb;
 }
 
```

### Patch 5

**Root cause:** The bug occurs because after wrapping the first line, the method prepends padding before recomputing the wrap position, so for long words or heavily indented continuation lines `findWrapPos` can keep returning the indentation boundary and the loop makes no progress.

**Why pre-patch overfits:** The pre-patch does address the infinite-loop symptom for the reported tests, but it does so with different semantics from the intended formatter behavior: it strips leading spaces from continuation text and asks `findWrapPos` to search starting at `nextLineTabStop` on text that no longer contains that padding. That can produce incorrect wrapping/alignment or spurious exceptions on other reasonable inputs, so it is not a generally correct fix for the root cause.

**OD refinement (NECESSARY):** The refinement removes the pre-patch's semantic change of trimming the remaining text and calling `findWrapPos` with `nextLineTabStop` as the start position; instead it keeps the original remaining text, computes wrapping from position 0, and only adds padding when appending. This is necessary because the pre-patch measures wrap positions on the unpadded text with a nonzero start offset, which changes wrapping behavior and can miss valid break points or trigger the same-position exception on inputs where the remaining text length is near the width boundary.

**Verdict:** Because the refinement made a NECESSARY semantic correction to how continuation lines are wrapped, the pre-patch was not already correct. It passed the tests with an altered, input-sensitive wrapping strategy rather than a proper general fix.

**Before OD (overfitting):**
```diff
@@ -19,10 +19,10 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
-pos = findWrapPos(text, width, 0);
+text = text.substring(pos).trim();
+pos = findWrapPos(text, width, nextLineTabStop);
 
 if (pos == -1)
 {
-sb.append(text);
+sb.append(padding).append(text);
 
 return sb;
@@ -33,5 +33,5 @@
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(padding).append(rtrim(text.substring(0, pos))).append(defaultNewLine);
 }
 }
```

**After OD (correct):**
```diff
@@ -2,4 +2,8 @@
 int nextLineTabStop, String text)
 {
+if (text == null) {
+return sb;
+}
+
 int pos = findWrapPos(text, width, 0);
 
@@ -19,10 +23,10 @@
 {
 int lastPos = pos;
-text = padding + text.substring(pos).trim();
+text = text.substring(pos);
 pos = findWrapPos(text, width, 0);
 
 if (pos == -1)
 {
-sb.append(text);
+sb.append(padding).append(rtrim(text));
 
 return sb;
@@ -33,5 +37,5 @@
 }
 
-sb.append(rtrim(text.substring(0, pos))).append(defaultNewLine);
+sb.append(padding).append(rtrim(text.substring(0, pos))).append(defaultNewLine);
 }
 }
```

**Ground truth fix:**
```diff
@@ -18,5 +18,4 @@
 while (true)
 {
-int lastPos = pos;
 text = padding + text.substring(pos).trim();
 pos = findWrapPos(text, width, 0);
@@ -27,8 +26,10 @@
 
 return sb;
-} else
-if (pos == lastPos)
-{
-throw new RuntimeException("Text too long for line - throwing exception to avoid infinite loop [CLI-162]: " + text);
+}
+
+if ( (text.length() > width) && (pos == nextLineTabStop - 1) ) {
+sb.append(text);
+
+return sb;
 }
 
```

---

## Cli-38

### Patch 1

**Root cause:** The bug is that isShortOption only checks whether the whole substring after '-' is a declared short option, so it misclassifies tokens that are either concatenated short options or valid multi-character short options with attached values.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause: it assumes every character after '-' must be an independent short option, which breaks valid multi-character short options that Commons CLI supports. For example, with a declared short option 't1', isShortOption('-t1') would return false unless both 't' and '1' were separately defined, so it would fail on reasonable inputs beyond the provided test scenario.

**OD refinement (NECESSARY):** The refinement restores parsing of the option name by stripping any '=value' suffix and, crucially, first checking whether the entire optName is itself a valid short option before falling back to per-character concatenation logic. This is necessary because the pre-patch removed support for legitimate multi-character short options such as '-t1' and also mishandles tokens containing '=' by iterating over non-option characters.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic that the pre-patch does not achieve by any alternative means, the original patch was a true overfitting patch rather than a genuinely correct fix.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,10 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
 // check for several concatenated short options
+for (int i = 1; i < token.length(); i++) {
+if (!options.hasShortOption(String.valueOf(token.charAt(i)))) {
+return false;
 }
+}
+return true;
+}
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 2

**Root cause:** The bug is that isShortOption only checked whether the whole token body was a declared short option and failed to recognize tokens that should be treated as concatenated short options while still preserving support for multi-character short option names like t1.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it treats every token after '-' as a sequence of one-character short options, so a valid short option like 't1' is rejected unless both 't' and '1' are separately defined. It would therefore fail on reasonable inputs beyond the shown test, including any legitimate multi-character short option or forms like '-t1=value'.

**OD refinement (NECESSARY):** The refinement restored the missing logic that first checks whether the entire option name (before any '=value') is itself a valid short option, and only falls back to per-character checking for concatenated options if that fails. This is necessary because the pre-patch incorrectly requires every character to be an individual short option, which breaks valid multi-character short options such as '-t1'.

**Verdict:** Because the refinement change is NECESSARY and adds logic absent from the pre-patch, the original patch was overfitting. It passed tests despite being semantically wrong for valid multi-character short options, so the refinement was required to make the fix genuinely correct.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,15 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+// handle concatenated short options
+for (int i = 1; i < token.length(); i++) {
+char c = token.charAt(i);
+if (c == '=') {
+break;
 }
+String optName = String.valueOf(c);
+if (!options.hasShortOption(optName)) {
+return false;
+}
+}
+return true;
+}
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 3

**Root cause:** The bug is that isShortOption only checked whether the whole token body after '-' was a declared short option, so it failed to recognize valid concatenated short options when the full substring was not itself an option name.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it treats every short-option token as a sequence of one-character options and never recognizes a valid multi-character short option as a single option. It would fail on reasonable inputs like '-t1' when 't1' is a defined short option but 't' and '1' are not both defined separately, so it is overfitting relative to the tested scenario.

**OD refinement (NECESSARY):** The refinement restored an essential semantic check that the pre-patch removed: first test whether the entire option name (before any '=value') is itself a valid short option, and only otherwise fall back to concatenated-character checking. This is necessary because Commons CLI allows multi-character short options such as 't1'; the pre-patch incorrectly requires every character to be an individual short option, which changes valid behavior.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic for whole-name short options, the pre-patch is semantically wrong on valid inputs beyond the failing test. Therefore the original patch was a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,10 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
 // check for several concatenated short options
+for (int i = 1; i < token.length(); i++) {
+char c = token.charAt(i);
+if (c == '=') break;
+if (!options.hasShortOption(String.valueOf(c))) return false;
 }
+return true;
+}
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 4

**Root cause:** The bug is that isShortOption only recognizes the whole post-dash token as a short option name and fails to also recognize tokens whose first character is a valid short option starting a concatenated short-option sequence.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it replaces whole-name recognition with per-character validation. It happens to pass the shown test only when all characters in a multi-character short option are also separately defined, but it would fail on reasonable inputs like a defined short option 't1' where '1' is not itself a short option, even though such tokens should be accepted.

**OD refinement (NECESSARY):** The refinement restores the missing check for whether the entire option name (before any '=value') is itself a valid short option, and only falls back to concatenation handling if that full-name check fails. This is necessary because the pre-patch incorrectly requires every character in the token to be an individually valid short option, which rejects legitimate multi-character short options such as '-t1' when 't1' is a defined short option.

**Verdict:** The refinement change is NECESSARY: it adds logic absent from the pre-patch that is required to recognize valid multi-character short options before treating a token as concatenated options. Therefore the original patch was overfitting rather than a semantically correct fix.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,19 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
+// check for several concatenated short options
+for (int i = 1; i < token.length(); i++) {
+char c = token.charAt(i);
+if (c == '=') {
+// handle the case where there's an '=' sign indicating a value
+String optName = token.substring(1, i);
 return options.hasShortOption(optName);
-// check for several concatenated short options
+} else {
+// check each character after '-' as a separate short option
+String optName = String.valueOf(c);
+if (!options.hasShortOption(optName)) {
+return false;
 }
+}
+}
+return true;
+}
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 5

**Root cause:** The bug is that isShortOption only checks whether the whole token body is a declared short option and therefore fails to correctly recognize tokens that may be either a multi-character short option or concatenated short options.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it rejects any short option whose name has multiple characters unless each character is itself a separate short option. For example, with a declared short option 't1', isShortOption("-t1") would return false if 't' and '1' are not both standalone options, so the patch would fail on reasonable inputs beyond the specific test configuration and is therefore overfitting.

**OD refinement (NECESSARY):** The refinement restored the missing logic to first test the entire option name (after stripping the leading '-' and any '=value') as a valid short option before falling back to concatenation handling. This is necessary because the pre-patch treats every character after '-' as an independent short option, which breaks valid multi-character short options such as '-t1' and also mishandles tokens containing '=value'.

**Verdict:** Because the refinement change is NECESSARY and adds logic absent from the pre-patch, the original patch was not semantically correct. It passed tests only under option sets where every character in the token happened to be recognized individually, which is not the intended behavior.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,11 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+// Check for concatenated short options
+for (int i = 1; i < token.length(); i++) {
+String optName = String.valueOf(token.charAt(i));
+if (!options.hasShortOption(optName)) {
+return false;
 }
+}
+return true;
+}
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 6

**Root cause:** The bug is that `isShortOption` only checks whether the whole token body is a declared short option, so it fails to recognize tokens that are actually concatenations of valid short options such as `-ab`.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause; it overgeneralizes by returning `true` for any token like `-xyz` as long as it has length greater than 2 and no `=`. That would mis-handle reasonable inputs such as undeclared options or mixed invalid concatenations (for example `-xz` when only `x` exists, or `-foo` with no such short options), so it passes the given test by coincidence rather than by correct option validation.

**OD refinement (NECESSARY):** The OD refinement replaced the pre-patch's unconditional `true` for any multi-character dash token without `=` with logic that first checks whether the full option name is a valid short option and otherwise validates concatenation against declared short options. This is necessary because the pre-patch classifies many invalid tokens as short options merely based on shape, not on whether the options actually exist.

**Verdict:** Because the refinement introduced necessary validation logic absent from the pre-patch, the original patch was a true overfitting fix. It solved the observed test by syntactic pattern matching instead of semantically checking declared short options.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,12 @@
 }
 
+// Check for concatenated options
+if (token.length() > 2 && !token.substring(1).contains("=")) {
+return true;
+}
+
 // remove leading "-" and "=value"
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
 return options.hasShortOption(optName);
-// check for several concatenated short options
 }
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

### Patch 7

**Root cause:** The bug is that isShortOption only checked the whole token body as one short option name and failed to recognize tokens that should be interpreted as concatenated short options, such as "-ab", while still preserving valid multi-character short options like "-t1".

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause. It over-approximates by accepting a token as a short option whenever any single character after '-' is a known option, so inputs like "-az" would be misclassified if only "a" exists, and its use of token.indexOf("") makes the final whole-name check always operate on an empty string. Thus it would fail on reasonable inputs beyond the triggering test and is not an alternative correct implementation.

**OD refinement (NECESSARY):** The refinement restored correct parsing of the option name by using token.indexOf("=") instead of token.indexOf("") and changed the logic to first test the full optName before considering concatenated short options. This is necessary because the pre-patch returns true if any character in the token is a known short option and, due to indexOf("") returning 0, computes an empty optName, breaking correct handling of valid multi-character short options and making the method accept many invalid tokens.

**Verdict:** The OD refinement made a necessary semantic correction, not a cosmetic change: it fixed broken optName extraction and required proper whole-name-first logic before concatenation handling. Therefore the original patch was a true overfitting patch rather than an already-correct fix.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,15 @@
 }
 
+// Check for concatenated short options
+for (int i = 1; i < token.length(); i++) {
+String optName = String.valueOf(token.charAt(i));
+if (options.hasShortOption(optName)) {
+return true;
+}
+}
+
 // remove leading "-" and "=value"
-int pos = token.indexOf("=");
+int pos = token.indexOf("");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
 return options.hasShortOption(optName);
-// check for several concatenated short options
 }
```

**After OD (correct):**
```diff
@@ -10,5 +10,16 @@
 int pos = token.indexOf("=");
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check if the entire optName is a valid short option
+if (options.hasShortOption(optName)) {
+return true;
 }
+
+// Check for several concatenated short options
+for (char c : optName.toCharArray()) {
+if (!options.hasShortOption(String.valueOf(c))) {
+return false;
+}
+}
+return true;
+}
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

## Cli-39

**Root cause:** The bug is that EXISTING_FILE_VALUE is handled like a plain File path, when it should attempt to open an existing file and return a FileInputStream (or otherwise fail during conversion) rather than returning a File object/string-like value.

**Why pre-patch overfits:** The pre-patch partially addresses the root cause by returning a FileInputStream for an existing file, but it does not implement the correct failure behavior for invalid inputs. In particular, for a nonexistent path it returns null rather than signaling a parse/conversion error, and it also does not guard against directories or access-denied cases, so it is not a fully correct implementation of EXISTING_FILE_VALUE semantics.

**OD refinement (NECESSARY):** The refinement changes the semantics for nonexistent and invalid paths: instead of returning null when the file does not exist, it throws ParseException, and it also rejects non-file paths and security failures explicitly. This is necessary because the pre-patch silently treats missing files as a successful null conversion, which is not the same behavior as the intended openFile-based fix and is incorrect for reasonable inputs beyond the provided tests.

**Verdict:** Because the OD refinement is NECESSARY and adds missing error-handling semantics that the pre-patch lacks, the pre-patch is a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -27,5 +27,14 @@
 else if (PatternOptionBuilder.EXISTING_FILE_VALUE == clazz)
 {
-return createFile(str);
+File file = new File(str);
+if (file.exists()) {
+try {
+return new FileInputStream(file);
+} catch (FileNotFoundException e) {
+throw new ParseException("File not found: " + str);
+}
+} else {
+return null;
+}
 }
 else if (PatternOptionBuilder.FILES_VALUE == clazz)
```

**After OD (correct):**
```diff
@@ -27,5 +27,18 @@
 else if (PatternOptionBuilder.EXISTING_FILE_VALUE == clazz)
 {
-return createFile(str);
+File file = new File(str);
+if (!file.exists()) {
+throw new ParseException("File does not exist: " + str);
+}
+if (!file.isFile()) {
+throw new ParseException("Path is not a file: " + str);
+}
+try {
+return new FileInputStream(file);
+} catch (FileNotFoundException e) {
+throw new ParseException("File not found: " + str);
+} catch (SecurityException e) {
+throw new ParseException("Access denied for file: " + str);
+}
 }
 else if (PatternOptionBuilder.FILES_VALUE == clazz)
```

**Ground truth fix:**
```diff
@@ -27,5 +27,5 @@
 else if (PatternOptionBuilder.EXISTING_FILE_VALUE == clazz)
 {
-return createFile(str);
+return openFile(str);
 }
 else if (PatternOptionBuilder.FILES_VALUE == clazz)
```

---

## Closure-128

**Root cause:** The bug is that isSimpleNumber incorrectly rejects the valid simple numeric string "0" by requiring the first character not be '0', instead of allowing the single-digit zero while still rejecting multi-digit numbers with leading zeros.

**Why pre-patch overfits:** The pre-patch fixes the observed failing test because it now accepts "0", but it overgeneralizes and treats all non-empty digit strings as simple numbers. That is semantically wrong for reasonable inputs such as "00" or "012", which should not be considered simple numbers and would be printed incorrectly as numeric literals rather than quoted property names.

**OD refinement (NECESSARY):** The OD refinement restored the leading-zero rule by changing the return condition from accepting any non-empty all-digit string to accepting only either a single digit or a number whose first digit is not '0'. It also added a null/empty guard, but the crucial semantic fix is reintroducing the distinction between "0" and strings like "00" or "012", which the pre-patch lost.

**Verdict:** Because the refinement change is NECESSARY and restores missing leading-zero logic that the pre-patch removed, the pre-patch does not correctly implement the root cause fix and is an overfitting patch.

**Before OD (overfitting):**
```diff
@@ -7,4 +7,4 @@
 }
 }
-return len > 0 && s.charAt(0) != '0';
+return len > 0;
 }
```

**After OD (correct):**
```diff
@@ -1,3 +1,6 @@
 static boolean isSimpleNumber(String s) {
+if (s == null || s.length() == 0) {
+return false;
+}
 int len = s.length();
 for (int index = 0; index < len; index++) {
@@ -7,4 +10,4 @@
 }
 }
-return len > 0 && s.charAt(0) != '0';
+return len > 0 && (s.charAt(0) != '0' || len == 1);
 }
```

**Ground truth fix:**
```diff
@@ -1,4 +1,7 @@
 static boolean isSimpleNumber(String s) {
 int len = s.length();
+if (len == 0) {
+return false;
+}
 for (int index = 0; index < len; index++) {
 char c = s.charAt(index);
@@ -7,4 +10,4 @@
 }
 }
-return len > 0 && s.charAt(0) != '0';
+return len == 1 || s.charAt(0) != '0';
 }
```

---

## Closure-6

**Root cause:** The bug was that assignment checking incorrectly special-cased constructor/enum mismatches by only registering a mismatch instead of emitting the normal type warning, thereby suppressing legitimate assignment errors.

**Why pre-patch overfits:** The pre-patch fixes the shown tests by forcing a warning for some constructor/enum mismatches, but it does not implement the real fix: it preserves the wrong special-case and adds an `equals` guard that can still skip diagnostics in cases where `canAssignTo` is false but the two `JSType` objects compare equal or otherwise fall into that branch. Reasonable inputs beyond the tests could therefore still miss required warnings, so the patch is overfitted rather than a correct root-cause fix.

**OD refinement (NECESSARY):** The refinement removed the constructor/enum special-case entirely, restoring uniform mismatch reporting whenever `canAssignTo` fails (except for the existing interface-method exemption in property assignments). In the pre-patch, constructor/enum pairs were still treated specially and only reported when `!leftType.equals(rightType)`, which changes semantics and can suppress real assignment warnings for incompatible constructor/enum-related types.

**Verdict:** Because the OD refinement made a necessary semantic correction?봢liminating the erroneous constructor/enum exception rather than tweaking it?봳he pre-patch did not fully fix the root cause. Its remaining special-case logic can still produce incorrect behavior outside the observed tests.

**Before OD (overfitting):**
```java
// --- L362 ---
boolean expectCanAssignToPropertyOf(NodeTraversal t, Node n, JSType rightType, JSType leftType, Node owner, String propName) {
    // The NoType check is a hack to make typedefs work OK.
    if (!leftType.isNoType() && !rightType.canAssignTo(leftType)) {
        if ((leftType.isConstructor() || leftType.isEnumType()) && (rightType.isConstructor() || rightType.isEnumType())) {
            if (!leftType.equals(rightType)) {
                registerMismatch(rightType, leftType, null);
                mismatch(t, n,
                    "assignment to property " + propName + " of " +
                    getReadableJSTypeName(owner, true),
                    rightType, leftType);
                return false;
            }
        } else {
            // Do not type-check interface methods, because we expect that
            // they will have dummy implementations that do not match the type
            // annotations.
            JSType ownerType = getJSType(owner);
            if (ownerType.isFunctionPrototypeType()) {
                FunctionType ownerFn = ownerType.toObjectType().getOwnerFunction();
                if (ownerFn.isInterface() &&
                    rightType.isFunctionType() && leftType.isFunctionType()) {
                    return true;
                }
            }
            mismatch(t, n,
                "assignment to property " + propName + " of " +
                getReadableJSTypeName(owner, true),
                rightType, leftType);
        }
        return false;
    }
    return true;
}

// ... 16 more lines
```

**After OD (correct):**
```java
// --- L362 ---
boolean expectCanAssignToPropertyOf(NodeTraversal t, Node n, JSType rightType, JSType leftType, Node owner, String propName) {
    if (!leftType.isNoType() && !rightType.canAssignTo(leftType)) {
        // Do not type-check interface methods, because we expect that
        // they will have dummy implementations that do not match the type
        // annotations.
        JSType ownerType = getJSType(owner);
        if (ownerType.isFunctionPrototypeType()) {
            FunctionType ownerFn = ownerType.toObjectType().getOwnerFunction();
            if (ownerFn.isInterface() &&
                rightType.isFunctionType() && leftType.isFunctionType()) {
                return true;
            }
        }
        mismatch(t, n,
            "assignment to property " + propName + " of " +
            getReadableJSTypeName(owner, true),
            rightType, leftType);
        return false;
    }
    return true;
}

// --- L402 ---
boolean expectCanAssignTo(NodeTraversal t, Node n, JSType rightType, JSType leftType, String msg) {
    if (!rightType.canAssignTo(leftType)) {
        mismatch(t, n, msg, rightType, leftType);
        return false;
    }
    return true;
}
```

---

## Closure-94

### Patch 1

**Root cause:** The bug is that isValidDefineValue recognizes only a few operators and mishandles operator arity, so valid define expressions using supported arithmetic/comparison/bitwise operators are rejected instead of being recursively validated by operand(s).

**Why pre-patch overfits:** The pre-patch fixes the specific failing ADD cases and also corrects the existing bitwise cases to validate both children, but it does not fully fix the underlying issue because other valid define expressions such as "1 - 2", "2 * 3", or "4 / 2" would still be rejected. Since the ground-truth fix requires broad operator coverage, the pre-patch remains incomplete and would fail on natural inputs beyond the provided tests.

**OD refinement (NECESSARY):** The OD refinement added support for additional binary operators (SUB, MUL, DIV) and introduced explicit child null checks for unary and binary cases. This is a real semantic expansion beyond the pre-patch, which only added ADD and still left other valid define expressions unhandled, so the refinement changed behavior on reasonable inputs outside the observed failing tests.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was overfitting to the tested addition scenario and did not fully repair the method's intended semantics.

**Before OD (overfitting):**
```diff
@@ -12,6 +12,8 @@
 case Token.BITOR:
 case Token.BITXOR:
+case Token.ADD: // Added to handle addition
+return isValidDefineValue(val.getFirstChild(), defines) && isValidDefineValue(val.getLastChild(), defines);
 
-// Uniary operators are valid if the child is valid.
+// Unary operators are valid if the child is valid.
 case Token.NOT:
 case Token.NEG:
```

**After OD (correct):**
```diff
@@ -12,9 +12,17 @@
 case Token.BITOR:
 case Token.BITXOR:
+case Token.ADD:
+case Token.SUB:
+case Token.MUL:
+case Token.DIV:
+Node leftChild = val.getFirstChild();
+Node rightChild = val.getLastChild();
+return leftChild != null && rightChild != null && isValidDefineValue(leftChild, defines) && isValidDefineValue(rightChild, defines);
 
-// Uniary operators are valid if the child is valid.
+// Unary operators are valid if the child is valid.
 case Token.NOT:
 case Token.NEG:
-return isValidDefineValue(val.getFirstChild(), defines);
+Node unaryChild = val.getFirstChild();
+return unaryChild != null && isValidDefineValue(unaryChild, defines);
 
 // Names are valid if and only if they are defines themselves.
```

**Ground truth fix:**
```diff
@@ -8,12 +8,31 @@
 
 // Binary operators are only valid if both children are valid.
+case Token.ADD:
 case Token.BITAND:
 case Token.BITNOT:
 case Token.BITOR:
 case Token.BITXOR:
+case Token.DIV:
+case Token.EQ:
+case Token.GE:
+case Token.GT:
+case Token.LE:
+case Token.LSH:
+case Token.LT:
+case Token.MOD:
+case Token.MUL:
+case Token.NE:
+case Token.RSH:
+case Token.SHEQ:
+case Token.SHNE:
+case Token.SUB:
+case Token.URSH:
+return isValidDefineValue(val.getFirstChild(), defines)
+&& isValidDefineValue(val.getLastChild(), defines);
 
 // Uniary operators are valid if the child is valid.
 case Token.NOT:
 case Token.NEG:
+case Token.POS:
 return isValidDefineValue(val.getFirstChild(), defines);
 
```

### Patch 2

**Root cause:** The bug is that isValidDefineValue treated several operators incorrectly: binary expressions were not validated by checking both operands, and valid define expressions should allow a broader set of unary/binary operators when their operands are themselves valid define values.

**Why pre-patch overfits:** The pre-patch fixes some failing cases, but it is semantically too restrictive: !DEF, !(1 & 2), or other valid define expressions under NOT would be rejected even though the intended rule is to accept NOT of any valid define value. It is also incomplete on binary operators compared with the actual fix, so it would fail on reasonable inputs beyond the provided tests.

**OD refinement (NECESSARY):** The OD refinement changed the semantics of unary NOT from a boolean-literal-only special case to the general rule used by the ground truth: any valid define value under NOT is allowed. It also broadened binary handling beyond the pre-patch's limited operator set by adding more arithmetic operators and using uniform child validation, which is required for correct behavior on valid define expressions outside the observed tests.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was overfitting. In particular, its special-casing of NOT and incomplete operator coverage do not correctly implement the root cause fix.

**Before OD (overfitting):**
```diff
@@ -12,7 +12,12 @@
 case Token.BITOR:
 case Token.BITXOR:
+case Token.ADD:
+case Token.SUB:
+return isValidDefineValue(val.getFirstChild(), defines) && isValidDefineValue(val.getLastChild(), defines);
 
-// Uniary operators are valid if the child is valid.
+// Unary operators are valid if the child is valid.
 case Token.NOT:
+Node child = val.getFirstChild();
+return child.getType() == Token.TRUE || child.getType() == Token.FALSE;
 case Token.NEG:
 return isValidDefineValue(val.getFirstChild(), defines);
```

**After OD (correct):**
```diff
@@ -12,9 +12,17 @@
 case Token.BITOR:
 case Token.BITXOR:
+case Token.ADD:
+case Token.SUB:
+case Token.MUL:
+case Token.DIV:
+Node leftChild = val.getFirstChild();
+Node rightChild = val.getLastChild();
+return leftChild != null && rightChild != null && isValidDefineValue(leftChild, defines) && isValidDefineValue(rightChild, defines);
 
-// Uniary operators are valid if the child is valid.
+// Unary operators are valid if the child is valid.
 case Token.NOT:
 case Token.NEG:
-return isValidDefineValue(val.getFirstChild(), defines);
+Node unaryChild = val.getFirstChild();
+return unaryChild != null && isValidDefineValue(unaryChild, defines);
 
 // Names are valid if and only if they are defines themselves.
```

**Ground truth fix:**
```diff
@@ -8,12 +8,31 @@
 
 // Binary operators are only valid if both children are valid.
+case Token.ADD:
 case Token.BITAND:
 case Token.BITNOT:
 case Token.BITOR:
 case Token.BITXOR:
+case Token.DIV:
+case Token.EQ:
+case Token.GE:
+case Token.GT:
+case Token.LE:
+case Token.LSH:
+case Token.LT:
+case Token.MOD:
+case Token.MUL:
+case Token.NE:
+case Token.RSH:
+case Token.SHEQ:
+case Token.SHNE:
+case Token.SUB:
+case Token.URSH:
+return isValidDefineValue(val.getFirstChild(), defines)
+&& isValidDefineValue(val.getLastChild(), defines);
 
 // Uniary operators are valid if the child is valid.
 case Token.NOT:
 case Token.NEG:
+case Token.POS:
 return isValidDefineValue(val.getFirstChild(), defines);
 
```

---

## Compress-28

### Patch 1

**Root cause:** The bug is that read() treated an unexpected end of the underlying stream during a tar entry as normal EOF instead of throwing an IOException for a truncated archive.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it assumes totalRead < numToRead implies truncation whenever entryOffset < entrySize. On reasonable inputs, especially streams that legally return short reads, it would throw IOException even though more data remains, so it is semantically wrong beyond the specific failing test.

**OD refinement (NECESSARY):** The refinement changed the truncation check from "any short read before entry end" to the correct condition: only when is.read returns -1 while the current entry still expects more bytes. This is necessary because InputStream.read is allowed to return fewer bytes than requested without indicating EOF, so the pre-patch would incorrectly throw on legitimate partial reads.

**Verdict:** The OD refinement was NECESSARY: it added the missing distinction between a short read and true EOF. Since the pre-patch lacks that required logic and would misbehave on valid short-read streams, it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || (totalRead < numToRead && entryOffset < entrySize)) {
 hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -17,4 +17,10 @@
 if (totalRead == -1) {
 hasHitEOF = true;
+} else if (totalRead < numToRead) {
+entryOffset += totalRead;
+if (entryOffset < entrySize) {
+hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
+}
 } else {
 entryOffset += totalRead;
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

**Root cause:** The method treats an unexpected end-of-stream while bytes are still expected for the current TAR entry as normal EOF instead of signaling a truncated archive via IOException.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it throws IOException whenever totalRead < numToRead, even if the read was a valid non-EOF short read or simply the final chunk that exactly completes the entry with fewer bytes than requested due to available() limits and stream behavior. Thus it can fail on reasonable inputs outside the triggering test, so it is an overfitting patch rather than an alternative correct implementation.

**OD refinement (NECESSARY):** The refinement changed the truncation condition from 'any short read' to 'a short read that still leaves the entry incomplete'. This is necessary because InputStream.read is allowed to return fewer bytes than requested without indicating EOF, so the pre-patch would incorrectly throw on legitimate partial reads and also mishandles state around that case.

**Verdict:** Because the OD refinement introduced necessary logic to distinguish true truncation from ordinary short reads, the pre-patch was not semantically correct. Its blanket 'short read means truncated' rule is stronger than the ground-truth behavior and would produce false exceptions on valid inputs.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || totalRead < numToRead) {
 hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -17,4 +17,10 @@
 if (totalRead == -1) {
 hasHitEOF = true;
+} else if (totalRead < numToRead) {
+entryOffset += totalRead;
+if (entryOffset < entrySize) {
+hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
+}
 } else {
 entryOffset += totalRead;
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

### Patch 3

**Root cause:** The bug is that an unexpected end-of-stream while there are still bytes remaining in the current TAR entry is treated as normal EOF instead of signaling a truncated archive with an IOException.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it assumes totalRead < numToRead implies truncation, which is not a valid assumption for general InputStream behavior. On reasonable inputs such as streams that legally return short reads, it would throw IOException even though the TAR entry is not truncated, so it is an overfitting patch despite passing the provided tests.

**OD refinement (NECESSARY):** The refinement changes the truncation condition from "any partial read smaller than requested" to the correct condition "underlying stream returned -1 before the entry's declared size was fully consumed." This is necessary because InputStream.read is allowed to return fewer bytes than requested without indicating truncation, so the pre-patch would incorrectly throw on legitimate short reads.

**Verdict:** Because the OD refinement introduced necessary missing logic for distinguishing true truncation from normal short reads, the pre-patch was semantically wrong on valid inputs beyond the test case. Therefore the original patch was a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || totalRead < numToRead) {
 hasHitEOF = true;
+throw new IOException("Truncated tar entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -15,5 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
+throw new IOException("Truncated TAR entry");
+} else if (totalRead == -1) {
 hasHitEOF = true;
 } else {
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

### Patch 4

**Root cause:** The method treats an unexpected end of the underlying stream during an entry as normal EOF instead of signaling a truncated TAR entry with an IOException.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it throws on any short read before updating entryOffset, even though short reads are legal and do not by themselves indicate truncation. On reasonable inputs such as streams that return partial data chunks, it would incorrectly throw IOException for non-truncated entries, so it is an overfitting fix rather than a semantically correct one.

**OD refinement (NECESSARY):** The refinement removes the pre-patch's assumption that any short read (bytesRead < numToRead) means truncation, and instead only throws after a short read when the entry is still incomplete. This is necessary because InputStream.read is allowed to return fewer bytes than requested without EOF, so the pre-patch can falsely reject valid archives.

**Verdict:** Because the OD refinement introduced necessary logic to distinguish legitimate short reads from true truncation, the pre-patch was not already correct. Its behavior would be wrong on valid partial-read streams, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -12,5 +12,9 @@
 numToRead = Math.min(numToRead, available());
 
-totalRead = is.read(buf, offset, numToRead);
+int bytesRead = is.read(buf, offset, numToRead);
+if (bytesRead < numToRead && entryOffset + bytesRead < entrySize) {
+throw new IOException("Truncated entry");
+}
+totalRead = bytesRead;
 count(totalRead);
 
```

**After OD (correct):**
```diff
@@ -17,4 +17,10 @@
 if (totalRead == -1) {
 hasHitEOF = true;
+} else if (totalRead < numToRead) {
+entryOffset += totalRead;
+if (entryOffset < entrySize) {
+hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
+}
 } else {
 entryOffset += totalRead;
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

### Patch 5

**Root cause:** The method treated an unexpected end-of-stream inside a TAR entry as normal EOF instead of throwing an IOException for a truncated archive.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it assumes bytesRead < numToRead implies truncation when entryOffset + bytesRead < entrySize. On reasonable valid inputs where the underlying stream legally returns a short positive read before more data remains, it would incorrectly throw IOException, so it is an overfitting patch rather than an alternative correct implementation.

**OD refinement (NECESSARY):** The refinement changed the truncation check from "fewer bytes than requested" to the correct condition of underlying read returning -1 before the entry has been fully consumed. This is necessary because InputStream.read is allowed to return any positive number smaller than the requested length without indicating truncation, so the pre-patch would wrongly throw on valid partial reads.

**Verdict:** Because the OD refinement introduced necessary semantic logic for detecting truncation correctly, the pre-patch was not already correct. Its short-read check is behaviorally wrong beyond the triggering test and can reject non-truncated archives.

**Before OD (overfitting):**
```diff
@@ -12,5 +12,10 @@
 numToRead = Math.min(numToRead, available());
 
-totalRead = is.read(buf, offset, numToRead);
+int bytesRead = is.read(buf, offset, numToRead);
+if (bytesRead < numToRead && entryOffset + bytesRead < entrySize) {
+throw new IOException("Truncated tar entry");
+}
+totalRead = bytesRead;
+
 count(totalRead);
 
```

**After OD (correct):**
```diff
@@ -15,7 +15,8 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated tar entry");
+} else if (totalRead != -1) {
 entryOffset += totalRead;
 }
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

## Compress-31

### Patch 1

**Root cause:** The bug is that parseOctal incorrectly treated an embedded NUL inside the significant octal digits as a terminator instead of rejecting it as invalid input.

**Why pre-patch overfits:** The pre-patch does address the original embedded-NUL bug by throwing instead of breaking, but it also adds incorrect behavior by rejecting any space encountered in the loop, even when the field is all spaces and should parse as 0 after leading-space skipping and trailing trimming. For example, an all-space field of valid tar header width would leave start == end in the ground-truth fix and return 0, while the pre-patch can leave one interior space and throw, so it is not semantically correct beyond the observed tests.

**OD refinement (NECESSARY):** The OD refinement removed the newly added special-case check that explicitly threw IllegalArgumentException for embedded NUL or space before the normal digit-range validation. This change is necessary because embedded spaces and NULs are already invalidated correctly by the existing range check, and the pre-patch's extra check changes behavior for valid edge cases such as a field consisting only of spaces after trimming.

**Verdict:** Because the refinement was NECESSARY and removed behavior that made the pre-patch reject reasonable valid inputs, the pre-patch was an overfitting fix rather than a genuinely correct one.

**Before OD (overfitting):**
```diff
@@ -33,6 +33,7 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
+if (currentByte == 0 || currentByte == ' ') {
+throw new IllegalArgumentException(
+exceptionMessage(buffer, offset, length, start, currentByte));
 }
 // CheckStyle:MagicNumber OFF
```

**After OD (correct):**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

**Ground truth fix:**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

### Patch 2

**Root cause:** The bug is that parseOctal incorrectly treats an embedded NUL inside the significant digit region as a terminator instead of rejecting it as an invalid octal character; the correct fix is to remove that special-case break and let normal validation throw.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause. Its added pre-scan runs before trimming trailing padding, so it throws on ordinary valid tar numeric fields ending in space or NUL, and its loop still breaks on NUL rather than treating it as invalid via the existing digit-range check. Thus it would mis-handle reasonable non-test inputs and is an overfitting patch.

**OD refinement (NECESSARY):** The OD refinement removed two behavioral changes from the pre-patch: the added pre-scan that rejected any space or NUL before trailing trimming, and the remaining in-loop special case that still broke on NUL. This was necessary because the pre-patch was semantically wrong: it rejects valid inputs with required trailing space/NUL padding and still preserves the original incorrect NUL-termination behavior in the main parsing loop.

**Verdict:** Because the refinement change is NECESSARY and removes genuinely incorrect logic from the pre-patch, the original patch was not a valid alternative fix. It passed tests despite breaking valid padded octal fields and retaining the wrong embedded-NUL handling path.

**Before OD (overfitting):**
```diff
@@ -18,4 +18,11 @@
 } else {
 break;
+}
+}
+
+// Check for embedded NULs or spaces before processing octal digits
+for (int i = start; i < end; i++) {
+if (buffer[i] == ' ' || buffer[i] == 0) {
+throw new IllegalArgumentException("Invalid character found in octal string");
 }
 }
```

**After OD (correct):**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

**Ground truth fix:**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

### Patch 3

**Root cause:** The bug is that parseOctal incorrectly treated an embedded NUL inside the significant octal digits as a terminator instead of rejecting it as invalid input, while a leading NUL field should still be interpreted as zero.

**Why pre-patch overfits:** The pre-patch does fix the embedded-NUL/embedded-space acceptance problem by rejecting those characters inside the parsed region, but it also removes valid behavior for zero-valued fields that begin with NUL. On reasonable inputs beyond the shown tests, such as a valid tar header field whose first byte is NUL, the pre-patch would incorrectly fail, so it does not fully implement the correct semantics.

**OD refinement (NECESSARY):** The refinement restored the special-case check `if (buffer[start] == 0) return 0L;` and removed the new rejection of any in-range NUL or space before digit validation. This is necessary because a tar numeric field consisting of a leading NUL is valid and should parse as zero, whereas the pre-patch would now throw an IllegalArgumentException for such inputs.

**Verdict:** Because the refinement made a NECESSARY semantic correction?봱einstating valid handling of leading-NUL zero fields?봳he pre-patch was not already correct. It passed tests with an over-restrictive behavior that breaks legitimate inputs, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -6,8 +6,4 @@
 if (length < 2){
 throw new IllegalArgumentException("Length "+length+" must be at least 2");
-}
-
-if (buffer[start] == 0) {
-return 0L;
 }
 
@@ -33,6 +29,7 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
+if (currentByte == 0 || currentByte == ' ') {
+throw new IllegalArgumentException(
+exceptionMessage(buffer, offset, length, start, currentByte));
 }
 // CheckStyle:MagicNumber OFF
```

**After OD (correct):**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

**Ground truth fix:**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

### Patch 4

**Root cause:** The bug is that parseOctal silently stops when it encounters an embedded NUL inside the significant octal field instead of rejecting any non-octal character after trimming only leading spaces and trailing space/NUL padding.

**Why pre-patch overfits:** The pre-patch does fix the original embedded-NUL bug, but it does so with stronger behavior than the ground-truth fix by treating any remaining space or NUL as a special hard error even in cases where the correct method should return 0 after trimming. For example, an all-spaces field of valid length leaves start == end after skipping leading spaces, but the pre-patch can also reject space/NUL patterns that the correct implementation would normalize via the existing trimming logic, so it is not semantically aligned with the true fix.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch's extra check that explicitly threw IllegalArgumentException for embedded spaces and NULs before the normal octal-range validation. This change is necessary because, after trimming trailing padding, any remaining embedded space or NUL is already invalid and should be handled by the existing generic non-octal check; the pre-patch changes semantics by rejecting cases the correct implementation accepts, most notably a field consisting only of spaces.

**Verdict:** Because the refinement change is NECESSARY, not cosmetic, the pre-patch was not already equivalent to the confirmed-correct implementation. It over-constrains accepted inputs relative to the true fix, so it is a true overfitting/semantically incorrect patch.

**Before OD (overfitting):**
```diff
@@ -33,8 +33,9 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
+// CheckStyle:MagicNumber OFF
+if (currentByte == ' ' || currentByte == 0){
+throw new IllegalArgumentException(
+exceptionMessage(buffer, offset, length, start, currentByte));
 }
-// CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
 throw new IllegalArgumentException(
```

**After OD (correct):**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

**Ground truth fix:**
```diff
@@ -33,7 +33,4 @@
 for ( ;start < end; start++) {
 final byte currentByte = buffer[start];
-if (currentByte == 0) {
-break;
-}
 // CheckStyle:MagicNumber OFF
 if (currentByte < '0' || currentByte > '7'){
```

---

## Compress-35

**Root cause:** The bug is that verifyCheckSum manually parses the checksum field incorrectly instead of using tar's octal parsing semantics for the checksum field.

**Why pre-patch overfits:** The pre-patch fixes the main checksum-parsing issue for normal valid tar headers by collecting octal digits from the checksum field and comparing against signed/unsigned sums, which is an alternative implementation of the ground-truth fix. However, it is not fully correct semantically because if the checksum field contains no octal digits or the header is null/too short, it throws rather than returning false, so it would fail on inputs beyond the provided test.

**OD refinement (NECESSARY):** The refinement adds defensive handling for invalid inputs: it returns false for null/too-short headers, for empty parsed checksum strings, and when octal parsing throws NumberFormatException. This changes runtime behavior because the pre-patch can throw exceptions on malformed but reasonable inputs, whereas checksum verification should simply report false.

**Verdict:** Because the OD refinement introduced necessary error-handling logic that was genuinely absent, the pre-patch is not semantically robust. Its success is limited to well-formed inputs, so it is a true overfitting patch under this assessment.

**Before OD (overfitting):**
```diff
@@ -4,12 +4,10 @@
 long signedSum = 0;
 
-int digits = 0;
+StringBuilder storedSumStr = new StringBuilder();
 for (int i = 0; i < header.length; i++) {
 byte b = header[i];
-if (CHKSUM_OFFSET  <= i && i < CHKSUM_OFFSET + CHKSUMLEN) {
-if ('0' <= b && b <= '7' && digits++ < 6) {
-storedSum = storedSum * 8 + b - '0';
-} else if (digits > 0) {
-digits = 6;
+if (CHKSUM_OFFSET <= i && i < CHKSUM_OFFSET + CHKSUMLEN) {
+if ('0' <= b && b <= '7') {
+storedSumStr.append((char) b);
 }
 b = ' ';
@@ -18,4 +16,5 @@
 signedSum += b;
 }
+storedSum = Long.parseLong(storedSumStr.toString().trim(), 8);
 return storedSum == unsignedSum || storedSum == signedSum;
 }
```

**After OD (correct):**
```diff
@@ -1,15 +1,17 @@
 public static boolean verifyCheckSum(byte[] header) {
+if (header == null || header.length < CHKSUM_OFFSET + CHKSUMLEN) {
+return false;
+}
+
 long storedSum = 0;
 long unsignedSum = 0;
 long signedSum = 0;
 
-int digits = 0;
+StringBuilder storedSumStr = new StringBuilder();
 for (int i = 0; i < header.length; i++) {
 byte b = header[i];
-if (CHKSUM_OFFSET  <= i && i < CHKSUM_OFFSET + CHKSUMLEN) {
-if ('0' <= b && b <= '7' && digits++ < 6) {
-storedSum = storedSum * 8 + b - '0';
-} else if (digits > 0) {
-digits = 6;
+if (CHKSUM_OFFSET <= i && i < CHKSUM_OFFSET + CHKSUMLEN) {
+if ('0' <= b && b <= '7') {
+storedSumStr.append((char) b);
 }
 b = ' ';
@@ -18,4 +20,16 @@
 signedSum += b;
 }
+
+String checksumStr = storedSumStr.toString().trim();
+if (checksumStr.isEmpty()) {
+return false;
+}
+
+try {
+storedSum = Long.parseLong(checksumStr, 8);
+} catch (NumberFormatException e) {
+return false;
+}
+
 return storedSum == unsignedSum || storedSum == signedSum;
 }
```

**Ground truth fix:**
```diff
@@ -1,4 +1,4 @@
 public static boolean verifyCheckSum(byte[] header) {
-long storedSum = 0;
+long storedSum = parseOctal(header, CHKSUM_OFFSET, CHKSUMLEN);
 long unsignedSum = 0;
 long signedSum = 0;
@@ -8,9 +8,4 @@
 byte b = header[i];
 if (CHKSUM_OFFSET  <= i && i < CHKSUM_OFFSET + CHKSUMLEN) {
-if ('0' <= b && b <= '7' && digits++ < 6) {
-storedSum = storedSum * 8 + b - '0';
-} else if (digits > 0) {
-digits = 6;
-}
 b = ' ';
 }
```

---

## Compress-39

### Patch 1

**Root cause:** The bug is that sanitize must truncate overly long sanitized names to the maximum length and indicate truncation with an ellipsis, rather than returning the full sanitized string.

**Why pre-patch overfits:** The pre-patch partially addresses the length limit, but it does not implement truncation correctly: it breaks at length 255 and then unconditionally replaces positions 252-254 with '.', even when the original input was exactly 255 characters long and should not be marked as truncated. Thus it would fail on reasonable inputs outside the provided test, such as any already-valid 255-character string, so it is an overfitting patch rather than a fully correct fix.

**OD refinement (NECESSARY):** The refinement changes the truncation point from 255 to 252 and appends "..." only when truncation is actually needed, instead of always overwriting the last three characters whenever the result length reaches 255. This is necessary because the pre-patch corrupts any non-truncated input whose sanitized length is exactly 255 by turning its last three valid characters into dots, which is semantically wrong.

**Verdict:** Because the OD refinement introduced necessary logic to distinguish true truncation from exact-length outputs, the pre-patch was not semantically correct. Its unconditional ellipsis-at-255 behavior changes valid outputs and therefore overfits the observed test scenario.

**Before OD (overfitting):**
```diff
@@ -9,8 +9,18 @@
 if (block != null && block != Character.UnicodeBlock.SPECIALS) {
 sb.append(c);
+if (sb.length() >= 255) { break; }
 continue;
 }
 }
 sb.append('?');
+if (sb.length() >= 255) { break; }
+}
+if (sb.length() > 255) {
+sb.setLength(255);
+}
+if (sb.length() == 255) {
+sb.setCharAt(254, '.');
+sb.setCharAt(253, '.');
+sb.setCharAt(252, '.');
 }
 return sb.toString();
```

**After OD (correct):**
```diff
@@ -1,3 +1,6 @@
 public static String sanitize(String s) {
+if (s == null) {
+return "";
+}
 final char[] chars = s.toCharArray();
 final int len = chars.length;
@@ -9,8 +12,19 @@
 if (block != null && block != Character.UnicodeBlock.SPECIALS) {
 sb.append(c);
+if (sb.length() >= 252) {
+sb.append("...");
+break;
+}
 continue;
 }
 }
 sb.append('?');
+if (sb.length() >= 252) {
+sb.append("...");
+break;
+}
+}
+if (sb.length() > 255) {
+sb.setLength(255);
 }
 return sb.toString();
```

**Ground truth fix:**
```diff
@@ -1,4 +1,10 @@
 public static String sanitize(String s) {
-final char[] chars = s.toCharArray();
+final char[] cs = s.toCharArray();
+final char[] chars = cs.length <= MAX_SANITIZED_NAME_LENGTH ? cs : Arrays.copyOf(cs, MAX_SANITIZED_NAME_LENGTH);
+if (cs.length > MAX_SANITIZED_NAME_LENGTH) {
+for (int i = MAX_SANITIZED_NAME_LENGTH - 3; i < MAX_SANITIZED_NAME_LENGTH; i++) {
+chars[i] = '.';
+}
+}
 final int len = chars.length;
 final StringBuilder sb = new StringBuilder();
```

### Patch 2

**Root cause:** The method failed to enforce the maximum sanitized name length by truncating inputs longer than 255 characters and replacing the last three characters of the truncated result with ellipsis.

**Why pre-patch overfits:** The pre-patch does address the failing test by returning the first 252 sanitized characters plus '...', but it does so only after processing the entire input and without truncating the source characters first. That can produce incorrect results on longer inputs containing control or special characters after position 255, where the correct implementation should ignore them entirely, so the pre-patch is not a fully correct fix of the root cause.

**OD refinement (NECESSARY):** The OD refinement added new runtime behavior: a null check and, more importantly, an early break when the builder reaches the maximum length so the method only sanitizes up to the allowed limit before appending an ellipsis. This is necessary because the pre-patch truncates after fully sanitizing the whole string, which can cut through the wrong position relative to the intended 255-character limit and does not match the ground-truth behavior of truncating the character array first.

**Verdict:** Because the refinement change is NECESSARY and adds missing length-limiting behavior during sanitization rather than merely reformatting the code, the pre-patch was overfitting to the observed test and was not semantically correct.

**Before OD (overfitting):**
```diff
@@ -13,5 +13,8 @@
 }
 sb.append('?');
+if (sb.length() > 255) {
+break;
 }
-return sb.toString();
 }
+return sb.length() > 255 ? sb.substring(0, 255) : sb.toString();
+}
```

**After OD (correct):**
```diff
@@ -1,6 +1,10 @@
 public static String sanitize(String s) {
+if (s == null) {
+return "";
+}
 final char[] chars = s.toCharArray();
 final int len = chars.length;
 final StringBuilder sb = new StringBuilder();
+final int maxLength = 255;
 for (int i = 0; i < len; i++) {
 final char c = chars[i];
@@ -13,5 +17,12 @@
 }
 sb.append('?');
+if (sb.length() >= maxLength) {
+break;
 }
+}
+if (sb.length() > maxLength - 3) {
+return sb.substring(0, maxLength - 3) + "...";
+} else {
 return sb.toString();
 }
+}
```

**Ground truth fix:**
```diff
@@ -1,4 +1,10 @@
 public static String sanitize(String s) {
-final char[] chars = s.toCharArray();
+final char[] cs = s.toCharArray();
+final char[] chars = cs.length <= MAX_SANITIZED_NAME_LENGTH ? cs : Arrays.copyOf(cs, MAX_SANITIZED_NAME_LENGTH);
+if (cs.length > MAX_SANITIZED_NAME_LENGTH) {
+for (int i = MAX_SANITIZED_NAME_LENGTH - 3; i < MAX_SANITIZED_NAME_LENGTH; i++) {
+chars[i] = '.';
+}
+}
 final int len = chars.length;
 final StringBuilder sb = new StringBuilder();
```

---

## Csv-15

**Root cause:** The bug was an overly broad first-character check in MINIMAL quote mode that quoted any non-ASCII-leading character at the start of a new record, instead of only quoting truly special leading characters that can break CSV parsing.

**Why pre-patch overfits:** The pre-patch does narrow the buggy condition enough to stop quoting the euro sign, so it fixes the reported test under the default format. However, it hardcodes ',' and '"' in a method that already derives delimiter and quote from the active CSVFormat, so it would miss required quoting for records starting with a non-default delimiter or quote character and is therefore configuration-specific overfitting.

**OD refinement (NECESSARY):** The OD refinement replaced hardcoded character literals in the new leading-character condition with configuration-aware values such as quoteChar and delimChar. This is semantically necessary because the pre-patch only works for the default RFC4180 settings (quote='"', delimiter=',') and becomes wrong for other valid CSVFormat configurations where the delimiter or quote character differs.

**Verdict:** Because the refinement change is NECESSARY rather than cosmetic, the pre-patch lacked required configuration-dependent logic. It passes existing tests but is semantically wrong for reasonable non-default CSVFormat inputs, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -37,5 +37,5 @@
 char c = value.charAt(pos);
 
-if (newRecord && (c < 0x20 || c > 0x21 && c < 0x23 || c > 0x2B && c < 0x2D || c > 0x7E)) {
+if (newRecord && (c < 0x20 || c == 0x22 || c == 0x2C || c == 0x0A || c == 0x0D)) {
 quote = true;
 } else if (c <= COMMENT) {
```

**After OD (correct):**
```diff
@@ -8,4 +8,8 @@
 final char delimChar = getDelimiter();
 final char quoteChar = getQuoteCharacter().charValue();
+final char lfChar = '\n';
+final char crChar = '\r';
+final char spChar = ' ';
+final char commentChar = COMMENT;
 
 QuoteMode quoteModePolicy = getQuoteMode();
@@ -37,7 +41,7 @@
 char c = value.charAt(pos);
 
-if (newRecord && (c < 0x20 || c > 0x21 && c < 0x23 || c > 0x2B && c < 0x2D || c > 0x7E)) {
+if (newRecord && (c < 0x20 || c == quoteChar || c == delimChar || c == lfChar || c == crChar)) {
 quote = true;
-} else if (c <= COMMENT) {
+} else if (c <= commentChar) {
 // Some other chars at the start of a value caused the parser to fail, so for now
 // encapsulate if we start in anything less than '#'. We are being conservative
@@ -47,5 +51,5 @@
 while (pos < end) {
 c = value.charAt(pos);
-if (c == LF || c == CR || c == quoteChar || c == delimChar) {
+if (c == lfChar || c == crChar || c == quoteChar || c == delimChar) {
 quote = true;
 break;
@@ -59,5 +63,5 @@
 // Some other chars at the end caused the parser to fail, so for now
 // encapsulate if we end in anything less than ' '
-if (c <= SP) {
+if (c <= spChar) {
 quote = true;
 }
```

**Ground truth fix:**
```diff
@@ -37,7 +37,5 @@
 char c = value.charAt(pos);
 
-if (newRecord && (c < 0x20 || c > 0x21 && c < 0x23 || c > 0x2B && c < 0x2D || c > 0x7E)) {
-quote = true;
-} else if (c <= COMMENT) {
+if (c <= COMMENT) {
 // Some other chars at the start of a value caused the parser to fail, so for now
 // encapsulate if we start in anything less than '#'. We are being conservative
```

---

## Csv-2

### Patch 1

**Root cause:** The method accesses values[index] without handling the case where a header name maps to an index outside the record's value array, so an inconsistent record throws ArrayIndexOutOfBoundsException instead of IllegalArgumentException.

**Why pre-patch overfits:** The pre-patch fixes the out-of-bounds case, but it incorrectly throws IllegalArgumentException whenever mapping.get(name) returns null. This breaks reasonable inputs beyond the failing test, such as requesting a non-existent header name, which should return null rather than signal an inconsistent record.

**OD refinement (NECESSARY):** The refinement changed the behavior for index == null from throwing IllegalArgumentException to returning null. That is a real semantic correction, because the original method's contract and the ground-truth fix preserve the existing behavior that unknown header names return null, while only out-of-bounds mapped indices should trigger IllegalArgumentException.

**Verdict:** Because the refinement introduced necessary missing logic for the null-index case, the pre-patch was not semantically correct. It overfit by converting all non-success cases into IllegalArgumentException instead of preserving the original null return for absent header names.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,8 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index >= 0 && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("Record is inconsistent");
 }
+}
```

**After OD (correct):**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index >= 0 && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("Index out of bounds for the given name: " + name);
+}
+}
```

**Ground truth fix:**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
+try {
 return index != null ? values[index.intValue()] : null;
+} catch (ArrayIndexOutOfBoundsException e) {
+throw new IllegalArgumentException(
+String.format(
+"Index for header '%s' is %d but CSVRecord only has %d values!",
+name, index.intValue(), values.length));
 }
+}
```

### Patch 2

**Root cause:** The bug occurs when a header name maps to an index outside the bounds of the record's values array, causing an ArrayIndexOutOfBoundsException instead of the intended IllegalArgumentException, while missing headers should still return null.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test by rejecting out-of-bounds indices, but it does not fully preserve the method's original contract because it throws IllegalArgumentException when mapping.get(name) returns null. On reasonable inputs where a caller asks for an unmapped header name, the buggy method and ground-truth fix return null, whereas the pre-patch would now fail incorrectly.

**OD refinement (NECESSARY):** The refinement added a distinct check for index == null to preserve the original API behavior of returning null when the requested header name is not present in the mapping. The pre-patch incorrectly merged the null-index case with the out-of-bounds case and threw IllegalArgumentException for both, so the refinement restored required semantics rather than merely changing style.

**Verdict:** Because the OD refinement was NECESSARY to reintroduce correct handling of missing headers, the pre-patch was not a complete semantic fix. It overfit the tested inconsistency case by turning a previously valid null-return path into an exception.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,6 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index < values.length) { return values[index.intValue()]; }
+else if (index != null) { throw new IllegalArgumentException("Record is inconsistent"); }
+else { return null; }
 }
```

**After OD (correct):**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index >= 0 && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("Index out of bounds for the given name: " + name);
+}
+}
```

**Ground truth fix:**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
+try {
 return index != null ? values[index.intValue()] : null;
+} catch (ArrayIndexOutOfBoundsException e) {
+throw new IllegalArgumentException(
+String.format(
+"Index for header '%s' is %d but CSVRecord only has %d values!",
+name, index.intValue(), values.length));
 }
+}
```

### Patch 3

**Root cause:** The bug is that get(String name) directly indexes into values using the header mapping without handling the case where the mapped index is outside the record's actual value array, which should be reported as an IllegalArgumentException rather than leaking an ArrayIndexOutOfBoundsException.

**Why pre-patch overfits:** The pre-patch only handles the inconsistent-record case partially; it incorrectly throws IllegalArgumentException whenever index is null, changing the established behavior for absent header names. Thus it does not fully fix the root cause and would fail on reasonable inputs where a caller asks for a non-existent column name and should receive null.

**OD refinement (NECESSARY):** The refinement changed the behavior for missing headers: when mapping.get(name) returns null, it now returns null instead of throwing IllegalArgumentException. This is semantically important because the original buggy method and the ground-truth fix both preserve null for unknown names, while only out-of-bounds mapped indices should trigger IllegalArgumentException.

**Verdict:** Because the refinement introduced necessary missing logic for the index == null case, the pre-patch was not semantically correct. It overfit to the failing test by converting all non-success cases into IllegalArgumentException instead of distinguishing missing headers from inconsistent records.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,5 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index < values.length) { return values[index.intValue()]; }
+else { throw new IllegalArgumentException("Record is inconsistent"); }
 }
```

**After OD (correct):**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index >= 0 && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("Index out of bounds for the given name: " + name);
+}
+}
```

**Ground truth fix:**
```diff
@@ -5,4 +5,11 @@
 }
 final Integer index = mapping.get(name);
+try {
 return index != null ? values[index.intValue()] : null;
+} catch (ArrayIndexOutOfBoundsException e) {
+throw new IllegalArgumentException(
+String.format(
+"Index for header '%s' is %d but CSVRecord only has %d values!",
+name, index.intValue(), values.length));
 }
+}
```

---

## JacksonCore-14

**Root cause:** The bug is that IOContext rejected releasing a different buffer of equal size by using <= instead of <, while it should only reject buffers smaller than the original, and the exception message should reflect that condition.

**Why pre-patch overfits:** The pre-patch only updates the error message to satisfy the observed test expectation, but leaves the core logic unchanged and still incorrectly throws for a different buffer of the same length. That means it would fail on reasonable inputs outside the given tests where releasing an equal-sized non-identical buffer should be accepted, so it does not truly fix the root cause.

**OD refinement (NECESSARY):** The OD refinement changed the release-validation logic from `toRelease.length <= src.length` to `toRelease.length < src.length`, which is the actual semantic fix required by the ground truth. It also changed the exception construction to include sizes, but the essential part is allowing equal-sized replacement buffers; without that, the pre-patch still enforces the wrong behavior.

**Verdict:** Because the refinement introduced a necessary semantic change to the comparison logic, the pre-patch was not equivalent to the correct fix. It passed tests by matching the expected message while preserving incorrect buffer-release behavior, so it is a true overfitting patch.

**Before OD (overfitting):**
```java
// --- L272 ---
protected final void _verifyRelease(byte[] toRelease, byte[] src) {
    // 07-Mar-2016, tatu: As per [core#255], only prevent shrinking of buffer
    if ((toRelease != src) && (toRelease.length <= src.length)) { throw wrongBuf(); }
}

// --- L277 ---
protected final void _verifyRelease(char[] toRelease, char[] src) {
    // 07-Mar-2016, tatu: As per [core#255], only prevent shrinking of buffer
    if ((toRelease != src) && (toRelease.length <= src.length)) { throw wrongBuf(); }
}

// --- L282 ---
private IllegalArgumentException wrongBuf() {
    // sanity check failed; trying to return different, smaller buffer.
    return new IllegalArgumentException("Trying to release buffer smaller than original");
}
```

**After OD (correct):**
```java
// --- L272 ---
protected final void _verifyRelease(byte[] toRelease, byte[] src) {
    // 07-Mar-2016, tatu: As per [core#255], only prevent shrinking of buffer
    if ((toRelease != src) && (toRelease.length < src.length)) { throw wrongBuf(toRelease.length, src.length); }
}

// --- L277 ---
protected final void _verifyRelease(char[] toRelease, char[] src) {
    // 07-Mar-2016, tatu: As per [core#255], only prevent shrinking of buffer
    if ((toRelease != src) && (toRelease.length < src.length)) { throw wrongBuf(toRelease.length, src.length); }
}

// --- L282 ---
private IllegalArgumentException wrongBuf(int releasedLength, int originalLength) {
    // sanity check failed; trying to return different, smaller buffer.
    return new IllegalArgumentException("Trying to release buffer of size " + releasedLength + " which is smaller than original size " + originalLength);
}
```

---

## JacksonCore-4

**Root cause:** The bug is that once the buffer reaches the intended maximum segment length, the code stops growing meaningfully and can even fail to increase size, whereas it should continue growing beyond that threshold at a slower rate.

**Why pre-patch overfits:** The pre-patch avoids the immediate non-growth failure by making lengths above or at MAX_SEGMENT_LEN grow by 1, so it passes the provided test. However, it does not implement the actual root-cause fix: for any large buffer beyond MAX_SEGMENT_LEN, expansion becomes pathologically linear instead of proportional, contradicting the intended semantics and the reference fix.

**OD refinement (NECESSARY):** The OD refinement changed the post-threshold growth policy from increasing by only 1 character when len >= MAX_SEGMENT_LEN to increasing by 25% of the current length. This is a real semantic change, not a refactoring: it restores the intended behavior described in the comment and matches the ground-truth fix's slower-but-still-proportional growth above MAX_SEGMENT_LEN.

**Verdict:** Because the refinement change is NECESSARY and adds missing growth logic that the pre-patch does not achieve by any equivalent means, the pre-patch is a true overfitting fix that only satisfies the tested monotonic-growth condition.

**Before OD (overfitting):**
```diff
@@ -5,5 +5,5 @@
 final int len = curr.length;
 // but above intended maximum, slow to increase by 25%
-int newLen = (len == MAX_SEGMENT_LEN) ? (MAX_SEGMENT_LEN+1) : Math.min(MAX_SEGMENT_LEN, len + (len >> 1));
+int newLen = (len >= MAX_SEGMENT_LEN) ? (len + 1) : Math.min(MAX_SEGMENT_LEN, len + (len >> 1));
 return (_currentSegment = Arrays.copyOf(curr, newLen));
 }
```

**After OD (correct):**
```diff
@@ -4,6 +4,11 @@
 // Let's grow by 50% by default
 final int len = curr.length;
-// but above intended maximum, slow to increase by 25%
-int newLen = (len == MAX_SEGMENT_LEN) ? (MAX_SEGMENT_LEN+1) : Math.min(MAX_SEGMENT_LEN, len + (len >> 1));
+int newLen;
+if (len < MAX_SEGMENT_LEN) {
+newLen = Math.min(MAX_SEGMENT_LEN, len + (len >> 1));
+} else {
+// Once we reach MAX_SEGMENT_LEN, grow by a fixed percentage, e.g., 25%
+newLen = len + (len >> 2);
+}
 return (_currentSegment = Arrays.copyOf(curr, newLen));
 }
```

**Ground truth fix:**
```diff
@@ -4,6 +4,9 @@
 // Let's grow by 50% by default
 final int len = curr.length;
+int newLen = len + (len >> 1);
 // but above intended maximum, slow to increase by 25%
-int newLen = (len == MAX_SEGMENT_LEN) ? (MAX_SEGMENT_LEN+1) : Math.min(MAX_SEGMENT_LEN, len + (len >> 1));
+if (newLen > MAX_SEGMENT_LEN) {
+newLen = len + (len >> 2);
+}
 return (_currentSegment = Arrays.copyOf(curr, newLen));
 }
```

---

## JacksonDatabind-99

**Root cause:** The bug is that ReferenceType.buildCanonicalName() forgets to append the closing '>' after the referenced type, producing malformed canonical names for all reference types.

**Why pre-patch overfits:** The pre-patch does not fully fix the root cause; it only adds the missing closing '>' for non-array referenced types and invents an unsupported exception for array types. Since ReferenceType canonical names should consistently include angle brackets regardless of whether the referenced type is an array, the pre-patch would produce incorrect results on reasonable inputs such as AtomicReference<String[]> or other reference types whose content type is an array.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch's special-case branch for array referenced types and instead always wraps a non-null referenced type in angle brackets, which matches the actual canonical-name format. This change is necessary because the pre-patch incorrectly omits '<' and '>' for array content types, yielding malformed canonical names like "AtomicReference[Ljava.lang.String;" instead of "AtomicReference<[Ljava.lang.String;>" (or equivalent array canonical form).

**Verdict:** Because the refinement change is NECESSARY and corrects genuinely missing logic in the pre-patch, the original patch was overfitting. It passed the given tests only because they exercised a non-array reference type, not because it implemented the true canonical-name construction rule.

**Before OD (overfitting):**
```diff
@@ -3,6 +3,11 @@
 StringBuilder sb = new StringBuilder();
 sb.append(_class.getName());
+if (_referencedType.isArrayType()) {
+sb.append(_referencedType.toCanonical());
+} else {
 sb.append('<');
 sb.append(_referencedType.toCanonical());
+sb.append('>');
+}
 return sb.toString();
 }
```

**After OD (correct):**
```diff
@@ -2,7 +2,12 @@
 {
 StringBuilder sb = new StringBuilder();
+if (_class != null) {
 sb.append(_class.getName());
+}
+if (_referencedType != null) {
 sb.append('<');
 sb.append(_referencedType.toCanonical());
+sb.append('>');
+}
 return sb.toString();
 }
```

**Ground truth fix:**
```diff
@@ -5,4 +5,5 @@
 sb.append('<');
 sb.append(_referencedType.toCanonical());
+sb.append('>');
 return sb.toString();
 }
```

---

## Jsoup-13

**Root cause:** Node.hasAttr(String) incorrectly treated keys with the "abs:" prefix as ordinary attribute names instead of recognizing them as requests for a resolvable absolute URL derived from the underlying attribute.

**Why pre-patch overfits:** The pre-patch fixes the tested cases for keys like "abs:href", but it does not fully implement the intended semantics of an absolute-attribute query because it accepts an empty attribute name after "abs:". On reasonable unseen input such as hasAttr("abs:"), it can incorrectly return true despite no corresponding attribute existing, so it is not a fully correct root-cause fix.

**OD refinement (NECESSARY):** The refinement added a guard that rejects the degenerate key "abs:" by requiring the suffix after the prefix to be non-empty before calling absUrl. This is a real semantic change: the pre-patch returns true for hasAttr("abs:") whenever the element has a non-empty base URI, because absUrl("") can resolve to that base URI, even though there is no actual attribute name after the prefix.

**Verdict:** Because the refinement change is NECESSARY and closes a real semantic hole not covered by the tests, the pre-patch was overfitting rather than already correct.

**Before OD (overfitting):**
```diff
@@ -2,4 +2,4 @@
 Validate.notNull(attributeKey);
 
-return attributes.hasKey(attributeKey);
+return attributeKey.startsWith("abs:") ? !absUrl(attributeKey.substring(4)).isEmpty() : attributes.hasKey(attributeKey);
 }
```

**After OD (correct):**
```diff
@@ -2,4 +2,9 @@
 Validate.notNull(attributeKey);
 
+if (attributeKey.startsWith("abs:")) {
+String url = attributeKey.substring(4);
+return !url.isEmpty() && !absUrl(url).isEmpty();
+}
+
 return attributes.hasKey(attributeKey);
 }
```

**Ground truth fix:**
```diff
@@ -2,4 +2,9 @@
 Validate.notNull(attributeKey);
 
+if (attributeKey.toLowerCase().startsWith("abs:")) {
+String key = attributeKey.substring("abs:".length());
+if (attributes.hasKey(key) && !absUrl(key).equals(""))
+return true;
+}
 return attributes.hasKey(attributeKey);
 }
```

---

## Jsoup-6

### Patch 1

**Root cause:** The bug is that Matcher.appendReplacement interprets replacement text specially, so decoded entity values containing '$' or backslashes must be escaped with Matcher.quoteReplacement before being inserted.

**Why pre-patch overfits:** The pre-patch does fix the immediate appendReplacement escaping problem, but it also alters the decoding condition from the original/GT behavior and thereby rejects all valid supplementary Unicode entities. On reasonable inputs outside the provided tests, numeric entities above 0xFFFF would remain escaped instead of being converted, so the patch is semantically incomplete and incorrect.

**OD refinement (NECESSARY):** Beyond the quoting fix already present in the pre-patch, the refinement added real semantic handling for null input, invalid numeric code points, and especially supplementary Unicode entities above 0xFFFF by emitting surrogate pairs instead of leaving them unchanged. This is not cosmetic: the pre-patch changed the condition to only decode entities <= 0xFFFF, so valid entities such as &#x1D11E; would no longer be unescaped correctly.

**Verdict:** The refinement is NECESSARY because it restores missing semantic behavior for valid non-BMP code points that the pre-patch broke by narrowing the condition to charval <= 0xFFFF. Therefore the pre-patch is a true overfitting patch rather than an already-correct fix.

**Before OD (overfitting):**
```diff
@@ -22,9 +22,9 @@
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1 && charval <= 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0)));
 }
 }
```

**After OD (correct):**
```diff
@@ -1,9 +1,8 @@
 static String unescape(String string) {
-if (!string.contains("&"))
+if (string == null || !string.contains("&"))
 return string;
 
-Matcher m = unescapePattern.matcher(string); // &(#(x|X)?([0-9a-fA-F]+)|[a-zA-Z]+);?
-StringBuffer accum = new StringBuffer(string.length()); // pity matcher can't use stringbuilder, avoid syncs
-// todo: replace m.appendReplacement with own impl, so StringBuilder and quoteReplacement not required
+Matcher m = unescapePattern.matcher(string);
+StringBuffer accum = new StringBuffer(string.length());
 
 while (m.find()) {
@@ -12,19 +11,30 @@
 if (num != null) {
 try {
-int base = m.group(2) != null ? 16 : 10; // 2 is hex indicator
-charval = Integer.valueOf(num, base);
+int base = (m.group(2) != null) ? 16 : 10;
+charval = Integer.parseInt(num, base);
+if (charval < 0 || charval > 0x10FFFF)
+charval = -1; // invalid Unicode code point
 } catch (NumberFormatException e) {
-} // skip
+// skip malformed numeric entity
+}
 } else {
 String name = m.group(1);
-if (full.containsKey(name))
+if (name != null && full.containsKey(name))
 charval = full.get(name);
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1) {
+if (charval <= 0xFFFF) {
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+// Handle supplementary characters (surrogate pairs)
+int highSurrogate = 0xD800 | ((charval - 0x10000) >> 10);
+int lowSurrogate = 0xDC00 | ((charval - 0x10000) & 0x3FF);
+String c = new String(new char[]{(char) highSurrogate, (char) lowSurrogate});
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
+}
+} else {
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // keep malformed entity unchanged
 }
 }
```

**Ground truth fix:**
```diff
@@ -24,7 +24,7 @@
 if (charval != -1 || charval > 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // replace with original string
 }
 }
```

### Patch 2

**Root cause:** The bug is that Matcher.appendReplacement interprets replacement text specially, so unescaped backslashes or dollar signs in decoded entity output or preserved original entity text trigger IllegalArgumentException unless the replacement is passed through Matcher.quoteReplacement.

**Why pre-patch overfits:** The pre-patch only fixes the successful-decoding branch by quoting the decoded character, but it does not fix the same appendReplacement hazard when the entity is not decoded and the original match is reinserted. It also changes the range check so that valid numeric entities above 0xFFFF are treated as invalid rather than decoded, so it would behave incorrectly on reasonable inputs such as supplementary-plane numeric character references.

**OD refinement (NECESSARY):** The refinement added quoting for the else-branch replacement (preserving the original matched entity) and also broadened valid numeric entity handling to supplementary Unicode code points instead of rejecting anything above 0xFFFF. This is semantically important because the pre-patch still passes raw m.group(0) to appendReplacement, which can fail for unknown or malformed entities containing '$' or '\', and it incorrectly leaves valid code points above 0xFFFF unchanged.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic rewrite, the pre-patch was not a complete semantic fix. It still mishandles some replacement strings and rejects valid non-BMP entities, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -22,6 +22,6 @@
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
-String c = Character.toString((char) charval);
+if (charval != -1 && charval <= 0xFFFF) { // within range
+String c = Matcher.quoteReplacement(Character.toString((char) charval));
 m.appendReplacement(accum, c);
 } else {
```

**After OD (correct):**
```diff
@@ -1,9 +1,8 @@
 static String unescape(String string) {
-if (!string.contains("&"))
+if (string == null || !string.contains("&"))
 return string;
 
-Matcher m = unescapePattern.matcher(string); // &(#(x|X)?([0-9a-fA-F]+)|[a-zA-Z]+);?
-StringBuffer accum = new StringBuffer(string.length()); // pity matcher can't use stringbuilder, avoid syncs
-// todo: replace m.appendReplacement with own impl, so StringBuilder and quoteReplacement not required
+Matcher m = unescapePattern.matcher(string);
+StringBuffer accum = new StringBuffer(string.length());
 
 while (m.find()) {
@@ -12,19 +11,30 @@
 if (num != null) {
 try {
-int base = m.group(2) != null ? 16 : 10; // 2 is hex indicator
-charval = Integer.valueOf(num, base);
+int base = (m.group(2) != null) ? 16 : 10;
+charval = Integer.parseInt(num, base);
+if (charval < 0 || charval > 0x10FFFF)
+charval = -1; // invalid Unicode code point
 } catch (NumberFormatException e) {
-} // skip
+// skip malformed numeric entity
+}
 } else {
 String name = m.group(1);
-if (full.containsKey(name))
+if (name != null && full.containsKey(name))
 charval = full.get(name);
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1) {
+if (charval <= 0xFFFF) {
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+// Handle supplementary characters (surrogate pairs)
+int highSurrogate = 0xD800 | ((charval - 0x10000) >> 10);
+int lowSurrogate = 0xDC00 | ((charval - 0x10000) & 0x3FF);
+String c = new String(new char[]{(char) highSurrogate, (char) lowSurrogate});
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
+}
+} else {
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // keep malformed entity unchanged
 }
 }
```

**Ground truth fix:**
```diff
@@ -24,7 +24,7 @@
 if (charval != -1 || charval > 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // replace with original string
 }
 }
```

### Patch 3

**Root cause:** The bug is that Matcher.appendReplacement interprets replacement strings specially, so unescaped decoded characters or original entity text containing '$' or '\' can trigger IllegalArgumentException unless passed through Matcher.quoteReplacement.

**Why pre-patch overfits:** The pre-patch only fixes the successful-decoding branch by quoting c, but leaves the fallback branch as m.appendReplacement(accum, m.group(0)), which is still unsafe. For reasonable inputs such as malformed or unknown entities containing '$' or '\' in the matched text, it can still throw or behave incorrectly, so it does not fully fix the appendReplacement root cause.

**OD refinement (NECESSARY):** The OD refinement added quoting in the else branch and also broadened handling for null input and supplementary Unicode code points. The quoting of m.group(0) is semantically necessary because unresolved entities like "$"-containing text or backslashes in the matched original string are still passed to appendReplacement, which also requires a quoted replacement string.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was still semantically incomplete. It passed the available tests but remained vulnerable in the unchanged fallback path, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -22,7 +22,7 @@
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1 && charval <= 0xFFFF) { // within range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
 m.appendReplacement(accum, m.group(0));
```

**After OD (correct):**
```diff
@@ -1,9 +1,8 @@
 static String unescape(String string) {
-if (!string.contains("&"))
+if (string == null || !string.contains("&"))
 return string;
 
-Matcher m = unescapePattern.matcher(string); // &(#(x|X)?([0-9a-fA-F]+)|[a-zA-Z]+);?
-StringBuffer accum = new StringBuffer(string.length()); // pity matcher can't use stringbuilder, avoid syncs
-// todo: replace m.appendReplacement with own impl, so StringBuilder and quoteReplacement not required
+Matcher m = unescapePattern.matcher(string);
+StringBuffer accum = new StringBuffer(string.length());
 
 while (m.find()) {
@@ -12,19 +11,30 @@
 if (num != null) {
 try {
-int base = m.group(2) != null ? 16 : 10; // 2 is hex indicator
-charval = Integer.valueOf(num, base);
+int base = (m.group(2) != null) ? 16 : 10;
+charval = Integer.parseInt(num, base);
+if (charval < 0 || charval > 0x10FFFF)
+charval = -1; // invalid Unicode code point
 } catch (NumberFormatException e) {
-} // skip
+// skip malformed numeric entity
+}
 } else {
 String name = m.group(1);
-if (full.containsKey(name))
+if (name != null && full.containsKey(name))
 charval = full.get(name);
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1) {
+if (charval <= 0xFFFF) {
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+// Handle supplementary characters (surrogate pairs)
+int highSurrogate = 0xD800 | ((charval - 0x10000) >> 10);
+int lowSurrogate = 0xDC00 | ((charval - 0x10000) & 0x3FF);
+String c = new String(new char[]{(char) highSurrogate, (char) lowSurrogate});
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
+}
+} else {
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // keep malformed entity unchanged
 }
 }
```

**Ground truth fix:**
```diff
@@ -24,7 +24,7 @@
 if (charval != -1 || charval > 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // replace with original string
 }
 }
```

### Patch 4

**Root cause:** The bug is that Matcher.appendReplacement interprets replacement strings specially, so unescaped decoded characters like "$" or "\\" (and unchanged entity text) can trigger regex replacement parsing errors unless passed through Matcher.quoteReplacement.

**Why pre-patch overfits:** The pre-patch only partially fixes the root cause: it correctly quotes decoded replacement characters in the success branch, but it does not quote m.group(0) in the fallback branch, so inputs like an unknown entity containing "$" would still be unsafe if preserved literally. More importantly, by changing the condition to charval != -1 && charval <= 0xFFFF, it stops decoding numeric entities above 0xFFFF at all, whereas the original intent and refined patch handle such code points; thus it is not a fully correct semantic fix.

**OD refinement (NECESSARY):** The OD refinement added several behavioral changes beyond the pre-patch: null-input handling, quoting the unchanged original match in the else branch, and support/validation for supplementary Unicode code points above 0xFFFF. These are semantic fixes, not cosmetic, because the pre-patch still leaves appendReplacement exposed to special replacement syntax when preserving an unrecognized entity, and it also regresses valid numeric entities above 0xFFFF by leaving them unchanged instead of decoding them.

**Verdict:** Because the refinement introduced necessary missing logic rather than mere refactoring, and the pre-patch both leaves one appendReplacement path unquoted and mishandles supplementary code points, the pre-patch is a true overfitting/incomplete fix rather than already-correct behavior.

**Before OD (overfitting):**
```diff
@@ -22,7 +22,7 @@
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1 && charval <= 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
 m.appendReplacement(accum, m.group(0));
```

**After OD (correct):**
```diff
@@ -1,9 +1,8 @@
 static String unescape(String string) {
-if (!string.contains("&"))
+if (string == null || !string.contains("&"))
 return string;
 
-Matcher m = unescapePattern.matcher(string); // &(#(x|X)?([0-9a-fA-F]+)|[a-zA-Z]+);?
-StringBuffer accum = new StringBuffer(string.length()); // pity matcher can't use stringbuilder, avoid syncs
-// todo: replace m.appendReplacement with own impl, so StringBuilder and quoteReplacement not required
+Matcher m = unescapePattern.matcher(string);
+StringBuffer accum = new StringBuffer(string.length());
 
 while (m.find()) {
@@ -12,19 +11,30 @@
 if (num != null) {
 try {
-int base = m.group(2) != null ? 16 : 10; // 2 is hex indicator
-charval = Integer.valueOf(num, base);
+int base = (m.group(2) != null) ? 16 : 10;
+charval = Integer.parseInt(num, base);
+if (charval < 0 || charval > 0x10FFFF)
+charval = -1; // invalid Unicode code point
 } catch (NumberFormatException e) {
-} // skip
+// skip malformed numeric entity
+}
 } else {
 String name = m.group(1);
-if (full.containsKey(name))
+if (name != null && full.containsKey(name))
 charval = full.get(name);
 }
 
-if (charval != -1 || charval > 0xFFFF) { // out of range
+if (charval != -1) {
+if (charval <= 0xFFFF) {
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+// Handle supplementary characters (surrogate pairs)
+int highSurrogate = 0xD800 | ((charval - 0x10000) >> 10);
+int lowSurrogate = 0xDC00 | ((charval - 0x10000) & 0x3FF);
+String c = new String(new char[]{(char) highSurrogate, (char) lowSurrogate});
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
+}
+} else {
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // keep malformed entity unchanged
 }
 }
```

**Ground truth fix:**
```diff
@@ -24,7 +24,7 @@
 if (charval != -1 || charval > 0xFFFF) { // out of range
 String c = Character.toString((char) charval);
-m.appendReplacement(accum, c);
+m.appendReplacement(accum, Matcher.quoteReplacement(c));
 } else {
-m.appendReplacement(accum, m.group(0));
+m.appendReplacement(accum, Matcher.quoteReplacement(m.group(0))); // replace with original string
 }
 }
```

---

## Jsoup-75

**Root cause:** The bug is that HTML boolean attributes should be collapsed not only when their value is null or exactly equal to the key, but also when they are present with an empty string value (and generally via the centralized boolean-attribute collapse rule).

**Why pre-patch overfits:** The pre-patch fixes the shown test by avoiding ="" for empty boolean attributes, but it does so by skipping serialization whenever val.isEmpty() && Attribute.isBooleanAttribute(key), rather than correctly defining when an attribute should collapse. It would produce incorrect output on reasonable inputs such as a non-boolean attribute with an empty string value (which should serialize as key="" but would be emitted as a bare key) and also misses case-insensitive boolean self-values like CHECKED="Checked" in HTML.

**OD refinement (NECESSARY):** The refinement changes the collapse condition itself to treat boolean attributes with empty values and case-insensitive key-equal values as collapsible in HTML syntax, and always emits an explicit value otherwise. The pre-patch instead suppresses output of the value in a separate inner condition, which changes behavior for non-boolean empty-string attributes and misses the more complete collapse semantics embodied by shouldCollapseAttribute.

**Verdict:** Because the OD refinement introduced necessary missing logic rather than a cosmetic rewrite, the pre-patch was overfitting. It passed the existing tests but did not correctly implement the general attribute-collapsing semantics required by the root cause.

**Before OD (overfitting):**
```diff
@@ -7,11 +7,13 @@
 accum.append(' ').append(key);
 
-// collapse checked=null, checked="", checked=checked; write out others
+// collapse checked=null, checked='', checked=checked; write out others
 if (!(out.syntax() == Document.OutputSettings.Syntax.html
 && (val == null || val.equals(key) && Attribute.isBooleanAttribute(key)))) {
-accum.append("=\"");
-Entities.escape(accum, val == null ? EmptyString : val, out, true, false, false);
+if (val != null && !(val.isEmpty() && Attribute.isBooleanAttribute(key))) {
+accum.append('=').append('"');
+Entities.escape(accum, val, out, true, false, false);
 accum.append('"');
 }
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -7,9 +7,10 @@
 accum.append(' ').append(key);
 
-// collapse checked=null, checked="", checked=checked; write out others
+// collapse checked=null, checked='', checked=checked; write out others
 if (!(out.syntax() == Document.OutputSettings.Syntax.html
-&& (val == null || val.equals(key) && Attribute.isBooleanAttribute(key)))) {
-accum.append("=\"");
-Entities.escape(accum, val == null ? EmptyString : val, out, true, false, false);
+&& Attribute.isBooleanAttribute(key)
+&& (val == null || val.isEmpty() || val.equalsIgnoreCase(key)))) {
+accum.append('=').append('"');
+Entities.escape(accum, val == null ? "" : val, out, true, false, false);
 accum.append('"');
 }
```

**Ground truth fix:**
```diff
@@ -8,6 +8,5 @@
 
 // collapse checked=null, checked="", checked=checked; write out others
-if (!(out.syntax() == Document.OutputSettings.Syntax.html
-&& (val == null || val.equals(key) && Attribute.isBooleanAttribute(key)))) {
+if (!Attribute.shouldCollapseAttribute(key, val, out)) {
 accum.append("=\"");
 Entities.escape(accum, val == null ? EmptyString : val, out, true, false, false);
```

---

## Jsoup-86

**Root cause:** The bug is that asXmlDeclaration() checks total child nodes on the parsed Document but then blindly accesses the first child element, even when the document has no element children or its first node is a non-Element node.

**Why pre-patch overfits:** The pre-patch avoids the observed IndexOutOfBoundsException by guarding against non-Element first child nodes, so it fixes the immediate crash. However, it does not fully fix the method's semantics: for inputs where parsing produces an XmlDeclaration node directly, it will incorrectly return null instead of the declaration, so it remains incomplete beyond the triggering test.

**OD refinement (NECESSARY):** The refinement adds handling for the case where the parsed document's first child node is already an XmlDeclaration, returning it directly before falling back to the Element-based reconstruction path. This is semantically important because the pre-patch only accepts Element as the first node and otherwise returns null, which can mis-handle valid parsed XML declaration nodes.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic for a real, reasonable input shape (first child is XmlDeclaration), the pre-patch was not a fully correct fix; it merely avoided the crash in the tested scenario.

**Before OD (overfitting):**
```diff
@@ -3,5 +3,5 @@
 Document doc = Jsoup.parse("<" + data.substring(1, data.length() -1) + ">", baseUri(), Parser.xmlParser());
 XmlDeclaration decl = null;
-if (doc.childNodeSize() > 0) {
+if (doc.childNodeSize() > 0 && doc.child(0) instanceof Element) {
 Element el = doc.child(0);
 decl = new XmlDeclaration(NodeUtils.parser(doc).settings().normalizeTag(el.tagName()), data.startsWith("!"));
```

**After OD (correct):**
```diff
@@ -3,6 +3,8 @@
 Document doc = Jsoup.parse("<" + data.substring(1, data.length() -1) + ">", baseUri(), Parser.xmlParser());
 XmlDeclaration decl = null;
-if (doc.childNodeSize() > 0) {
-Element el = doc.child(0);
+if (doc.childNodeSize() > 0 && doc.childNode(0) instanceof XmlDeclaration) {
+decl = (XmlDeclaration) doc.childNode(0);
+} else if (doc.childNodeSize() > 0 && doc.childNode(0) instanceof Element) {
+Element el = (Element) doc.childNode(0);
 decl = new XmlDeclaration(NodeUtils.parser(doc).settings().normalizeTag(el.tagName()), data.startsWith("!"));
 decl.attributes().addAll(el.attributes());
```

**Ground truth fix:**
```diff
@@ -3,5 +3,5 @@
 Document doc = Jsoup.parse("<" + data.substring(1, data.length() -1) + ">", baseUri(), Parser.xmlParser());
 XmlDeclaration decl = null;
-if (doc.childNodeSize() > 0) {
+if (doc.children().size() > 0) {
 Element el = doc.child(0);
 decl = new XmlDeclaration(NodeUtils.parser(doc).settings().normalizeTag(el.tagName()), data.startsWith("!"));
```

---

## JxPath-8

### Patch 1

**Root cause:** The bug is that relational comparison logic treats NaN like an ordinary number and maps it to a comparison result, whereas any comparison involving NaN should return false in this method.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause: it detects NaN but then treats any NaN-involving comparison as comparison result 0. That means it will still produce incorrect results on reasonable inputs beyond the first failing assertion, notably equality comparisons such as `$nan = $nan`, which should be false under the confirmed-correct behavior but would evaluate to true in the pre-patch.

**OD refinement (NECESSARY):** The OD refinement changed the NaN handling from `return evaluateCompare(0)` to `return false` whenever either operand is NaN. This is semantically necessary because `evaluateCompare(0)` makes equality-style operations behave as if the operands were equal, so the pre-patch incorrectly returns true for cases like `NaN = NaN`, contrary to the intended fix implemented in the confirmed-correct patch.

**Verdict:** Because the refinement introduced a necessary semantic change in how NaN cases are handled, the pre-patch was not equivalent to the correct fix. It overfits by passing tests despite preserving wrong behavior for NaN equality-style comparisons.

**Before OD (overfitting):**
```diff
@@ -18,6 +18,9 @@
 return containsMatch((Iterator) right, left);
 }
-double ld = InfoSetUtil.doubleValue(left);
-double rd = InfoSetUtil.doubleValue(right);
+double ld = Double.isNaN(InfoSetUtil.doubleValue(left)) ? Double.NaN : InfoSetUtil.doubleValue(left);
+double rd = Double.isNaN(InfoSetUtil.doubleValue(right)) ? Double.NaN : InfoSetUtil.doubleValue(right);
+if (Double.isNaN(ld) || Double.isNaN(rd)) {
+return evaluateCompare(0);
+}
 return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
 }
```

**After OD (correct):**
```diff
@@ -18,6 +18,18 @@
 return containsMatch((Iterator) right, left);
 }
+if (left == null || right == null) {
+return evaluateCompare(0);
+}
 double ld = InfoSetUtil.doubleValue(left);
 double rd = InfoSetUtil.doubleValue(right);
-return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
+if (Double.isNaN(ld) || Double.isNaN(rd)) {
+return false;
 }
+if (ld == rd) {
+return evaluateCompare(0);
+} else if (ld < rd) {
+return evaluateCompare(-1);
+} else {
+return evaluateCompare(1);
+}
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,11 @@
 }
 double ld = InfoSetUtil.doubleValue(left);
+if (Double.isNaN(ld)) {
+return false;
+}
 double rd = InfoSetUtil.doubleValue(right);
+if (Double.isNaN(rd)) {
+return false;
+}
 return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
 }
```

### Patch 2

**Root cause:** The bug is that relational comparison logic treats NaN like an ordinary number, so the fallback comparison expression maps NaN cases to a positive comparison result instead of making any NaN-involved relational expression evaluate to false.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause. By converting any NaN case into `evaluateCompare(0)`, it makes expressions handled as equality compare as equal, so cases like `$nan = $nan` become true for the wrong reason and other NaN-involved equality cases can also be misclassified; the ground-truth fix instead returns false immediately for NaN in this method.

**OD refinement (NECESSARY):** The OD refinement changed NaN handling from `return evaluateCompare(0)` to `return false` when either operand is NaN, and also added a null guard. This is necessary because `evaluateCompare(0)` is operator-dependent: it makes equality expressions true, whereas the correct semantics for this relational-expression code path are that any comparison involving NaN should return false.

**Verdict:** Because the refinement introduced necessary semantic logic rather than a cosmetic rewrite, the pre-patch was overfitting. Its NaN handling is not equivalent to the correct fix and would produce incorrect results on reasonable inputs governed by this method's comparison semantics.

**Before OD (overfitting):**
```diff
@@ -20,4 +20,7 @@
 double ld = InfoSetUtil.doubleValue(left);
 double rd = InfoSetUtil.doubleValue(right);
+if (Double.isNaN(ld) || Double.isNaN(rd)) {
+return evaluateCompare(0);
+}
 return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
 }
```

**After OD (correct):**
```diff
@@ -18,6 +18,18 @@
 return containsMatch((Iterator) right, left);
 }
+if (left == null || right == null) {
+return evaluateCompare(0);
+}
 double ld = InfoSetUtil.doubleValue(left);
 double rd = InfoSetUtil.doubleValue(right);
-return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
+if (Double.isNaN(ld) || Double.isNaN(rd)) {
+return false;
 }
+if (ld == rd) {
+return evaluateCompare(0);
+} else if (ld < rd) {
+return evaluateCompare(-1);
+} else {
+return evaluateCompare(1);
+}
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,11 @@
 }
 double ld = InfoSetUtil.doubleValue(left);
+if (Double.isNaN(ld)) {
+return false;
+}
 double rd = InfoSetUtil.doubleValue(right);
+if (Double.isNaN(rd)) {
+return false;
+}
 return evaluateCompare(ld == rd ? 0 : ld < rd ? -1 : 1);
 }
```

---

## Lang-26

### Patch 1

**Root cause:** The bug is caused by creating the formatting Calendar without the formatter's locale, so locale-dependent week-numbering rules such as first day of week and minimal days in first week are taken from the default locale instead of mLocale.

**Why pre-patch overfits:** The pre-patch only forces Monday as first day of week and 4 minimal days, which happens to match Swedish week rules and makes the failing test pass, but it ignores mLocale entirely. It would produce incorrect results for locales whose week definitions differ from ISO-style settings, so it does not correctly fix the general root cause.

**OD refinement (NECESSARY):** The refinement replaced hardcoded ISO-like week settings on a default-locale GregorianCalendar with locale-based calendar construction using mLocale and mTimeZone. This is necessary because the correct behavior is not universally Monday/4-day-week; it must follow the formatter's locale for all locale-sensitive fields, not just the specific Swedish test case.

**Verdict:** Because the refinement change is NECESSARY and introduces missing locale-driven semantics, the pre-patch is an overfitting fix. It passes the observed test by hardcoding Swedish/ISO week rules rather than correctly honoring the formatter locale.

**Before OD (overfitting):**
```diff
@@ -1,4 +1,6 @@
 public String format(Date date) {
 Calendar c = new GregorianCalendar(mTimeZone);
+c.setMinimalDaysInFirstWeek(4);
+c.setFirstDayOfWeek(Calendar.MONDAY);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

**After OD (correct):**
```diff
@@ -1,4 +1,8 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+if (date == null) {
+throw new IllegalArgumentException("Date must not be null");
+}
+Calendar c = new GregorianCalendar(mLocale);
+c.setTimeZone(mTimeZone);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

**Ground truth fix:**
```diff
@@ -1,4 +1,4 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+Calendar c = new GregorianCalendar(mTimeZone, mLocale);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

### Patch 2

**Root cause:** The bug is caused by creating the Calendar without the formatter's locale, so locale-dependent week rules (first day of week and minimal days in first week) default incorrectly and produce wrong week numbers.

**Why pre-patch overfits:** The pre-patch does not fix the root cause; it bypasses it by hardcoding ISO-8601 week settings instead of deriving them from the formatter locale. It will give incorrect results for reasonable non-ISO locales, since FastDateFormat is supposed to format according to mLocale, not always Monday as first day and 4 minimal days.

**OD refinement (NECESSARY):** The refinement replaces hardcoded ISO week settings with locale-based calendar construction by using the formatter's locale and timezone, which is the actual source of the correct week-rule behavior. This is necessary because week numbering is locale-dependent in general, and forcing Monday/4 only matches some locales such as sv_SE rather than correctly honoring mLocale for all cases.

**Verdict:** Because the refinement change is NECESSARY and introduces the missing locale-sensitive calendar semantics, the pre-patch is a true overfitting fix that only happens to satisfy the observed test case.

**Before OD (overfitting):**
```diff
@@ -1,4 +1,6 @@
 public String format(Date date) {
 Calendar c = new GregorianCalendar(mTimeZone);
+c.setFirstDayOfWeek(Calendar.MONDAY);
+c.setMinimalDaysInFirstWeek(4);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

**After OD (correct):**
```diff
@@ -1,4 +1,8 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+if (date == null) {
+throw new IllegalArgumentException("Date must not be null");
+}
+Calendar c = new GregorianCalendar(mLocale);
+c.setTimeZone(mTimeZone);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

**Ground truth fix:**
```diff
@@ -1,4 +1,4 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+Calendar c = new GregorianCalendar(mTimeZone, mLocale);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

### Patch 3

**Root cause:** The bug is that format(Date) creates a GregorianCalendar with only the time zone and not the formatter's locale, so locale-dependent week rules (first day of week and minimal days in first week) are taken from the JVM default locale instead of mLocale.

**Why pre-patch overfits:** The pre-patch fixes the specific Swedish test because Sweden uses Monday as first day of week and 4 minimal days in the first week, but it does not fix the actual root cause of ignoring mLocale. For other locales whose week rules differ, it would compute week numbers incorrectly, so it is an overfitting patch rather than a general semantic fix.

**OD refinement (NECESSARY):** The refinement replaces hardcoded ISO-style week settings with locale-based calendar construction, using the formatter's locale and then applying the configured time zone. This is necessary because week numbering rules are locale-dependent in general, and the pre-patch forces Monday/4-day rules for every locale rather than honoring mLocale.

**Verdict:** Because the refinement change is NECESSARY and introduces the missing locale-sensitive calendar semantics, the pre-patch is not an alternative correct fix. It hardcodes one locale's week rules and would fail on reasonable inputs outside the observed test.

**Before OD (overfitting):**
```diff
@@ -2,4 +2,6 @@
 Calendar c = new GregorianCalendar(mTimeZone);
 c.setTime(date);
+c.setMinimalDaysInFirstWeek(4);
+c.setFirstDayOfWeek(Calendar.MONDAY);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
 }
```

**After OD (correct):**
```diff
@@ -1,4 +1,8 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+if (date == null) {
+throw new IllegalArgumentException("Date must not be null");
+}
+Calendar c = new GregorianCalendar(mLocale);
+c.setTimeZone(mTimeZone);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

**Ground truth fix:**
```diff
@@ -1,4 +1,4 @@
 public String format(Date date) {
-Calendar c = new GregorianCalendar(mTimeZone);
+Calendar c = new GregorianCalendar(mTimeZone, mLocale);
 c.setTime(date);
 return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
```

---

## Lang-54

### Patch 1

**Root cause:** The bug is that toLocale wrongly assumes that after the language and first underscore there must be a two-letter uppercase country code, so it rejects valid locales with an empty country and a variant such as "fr__POSIX".

**Why pre-patch overfits:** The pre-patch does handle the specific missing-country case like "fr__POSIX", but it overgeneralizes by accepting invalid inputs such as "en_USA", "en_A", or other malformed country segments that should still trigger IllegalArgumentException under the intended contract. Thus it does not fully preserve the method's required validation semantics and would behave incorrectly on reasonable inputs beyond the failing test.

**OD refinement (NECESSARY):** The refinement reintroduced format validation for the locale components: it enforces a two-letter lowercase language, allows an empty country, and requires any non-empty country to be exactly two uppercase letters. This is necessary because the pre-patch removed the original structural checks and accepted many malformed inputs by simply splitting on underscores and passing arbitrary substrings to Locale.

**Verdict:** Because the OD refinement was NECESSARY to restore missing validation logic, the pre-patch was not semantically correct. It fixed the observed test by loosening parsing too much rather than correctly implementing the locale format rules.

**Before OD (overfitting):**
```diff
@@ -4,5 +4,5 @@
 }
 int len = str.length();
-if (len != 2 && len != 5 && len < 7) {
+if (len < 2) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
@@ -18,16 +18,9 @@
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-char ch3 = str.charAt(3);
-char ch4 = str.charAt(4);
-if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-if (len == 5) {
-return new Locale(str.substring(0, 2), str.substring(3, 5));
+int variantIndex = str.indexOf('_', 3);
+if (variantIndex == -1) {
+return new Locale(str.substring(0, 2), str.substring(3));
 } else {
-if (str.charAt(5) != '_') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-return new Locale(str.substring(0, 2), str.substring(3, 5), str.substring(6));
+return new Locale(str.substring(0, 2), str.substring(3, variantIndex), str.substring(variantIndex + 1));
 }
 }
```

**After OD (correct):**
```diff
@@ -3,32 +3,27 @@
 return null;
 }
-int len = str.length();
-if (len != 2 && len != 5 && len < 7) {
+
+// Regular expression to match locale format: language[_country[_variant]]
+String[] parts = str.split("_", 3);
+if (parts.length < 1 || parts.length > 3) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-char ch0 = str.charAt(0);
-char ch1 = str.charAt(1);
-if (ch0 < 'a' || ch0 > 'z' || ch1 < 'a' || ch1 > 'z') {
+
+String language = parts[0];
+String country = parts.length >= 2 ? parts[1] : "";
+String variant = parts.length == 3 ? parts[2] : "";
+
+// Validate language code
+if (!language.matches("[a-z]{2}")) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-if (len == 2) {
-return new Locale(str, "");
-} else {
-if (str.charAt(2) != '_') {
+
+// Validate country code
+if (!country.isEmpty() && !country.matches("[A-Z]{2}")) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-char ch3 = str.charAt(3);
-char ch4 = str.charAt(4);
-if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
+
+// Variant can be any string, no specific validation needed
+
+return new Locale(language, country, variant);
 }
-if (len == 5) {
-return new Locale(str.substring(0, 2), str.substring(3, 5));
-} else {
-if (str.charAt(5) != '_') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-return new Locale(str.substring(0, 2), str.substring(3, 5), str.substring(6));
-}
-}
-}
```

**Ground truth fix:**
```diff
@@ -19,4 +19,7 @@
 }
 char ch3 = str.charAt(3);
+if (ch3 == '_') {
+return new Locale(str.substring(0, 2), "", str.substring(4));
+}
 char ch4 = str.charAt(4);
 if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
```

### Patch 2

**Root cause:** The bug is that toLocale rejects locales of the form language__variant (empty country with non-empty variant), such as "fr__POSIX", because it always requires positions 3 and 4 to be uppercase country letters.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test by allowing split-based parsing of "fr__POSIX", but it removes essential format validation for the country part and weakens the length checks. As a result, it would incorrectly accept many malformed locale strings outside the test suite, so it does not correctly preserve the original contract while fixing the root cause.

**OD refinement (NECESSARY):** The OD refinement added validation of the country component so that a non-empty country must match exactly two uppercase letters, instead of blindly accepting whatever appears between underscores. This is necessary because the pre-patch's split-based parsing accepts invalid inputs like "fr_1" or "fr_abc" as Locale objects, which violates the method's intended strict format checking.

**Verdict:** Because the refinement change is NECESSARY and restores missing semantic validation that the pre-patch dropped, the original patch was overfitting rather than a genuinely correct alternative fix.

**Before OD (overfitting):**
```diff
@@ -4,5 +4,5 @@
 }
 int len = str.length();
-if (len != 2 && len != 5 && len < 7) {
+if (len != 2 && len < 5) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
@@ -12,23 +12,12 @@
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-if (len == 2) {
-return new Locale(str, "");
-} else {
-if (str.charAt(2) != '_') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-char ch3 = str.charAt(3);
-char ch4 = str.charAt(4);
-if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-if (len == 5) {
-return new Locale(str.substring(0, 2), str.substring(3, 5));
-} else {
-if (str.charAt(5) != '_') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-return new Locale(str.substring(0, 2), str.substring(3, 5), str.substring(6));
+String[] parts = str.split("_", 3);
+switch (parts.length) {
+case 1:
+return new Locale(parts[0], "");
+case 2:
+return new Locale(parts[0], parts[1]);
+default:
+return new Locale(parts[0], parts[1], parts[2]);
 }
 }
-}
```

**After OD (correct):**
```diff
@@ -3,32 +3,27 @@
 return null;
 }
-int len = str.length();
-if (len != 2 && len != 5 && len < 7) {
+
+// Regular expression to match locale format: language[_country[_variant]]
+String[] parts = str.split("_", 3);
+if (parts.length < 1 || parts.length > 3) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-char ch0 = str.charAt(0);
-char ch1 = str.charAt(1);
-if (ch0 < 'a' || ch0 > 'z' || ch1 < 'a' || ch1 > 'z') {
+
+String language = parts[0];
+String country = parts.length >= 2 ? parts[1] : "";
+String variant = parts.length == 3 ? parts[2] : "";
+
+// Validate language code
+if (!language.matches("[a-z]{2}")) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-if (len == 2) {
-return new Locale(str, "");
-} else {
-if (str.charAt(2) != '_') {
+
+// Validate country code
+if (!country.isEmpty() && !country.matches("[A-Z]{2}")) {
 throw new IllegalArgumentException("Invalid locale format: " + str);
 }
-char ch3 = str.charAt(3);
-char ch4 = str.charAt(4);
-if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
+
+// Variant can be any string, no specific validation needed
+
+return new Locale(language, country, variant);
 }
-if (len == 5) {
-return new Locale(str.substring(0, 2), str.substring(3, 5));
-} else {
-if (str.charAt(5) != '_') {
-throw new IllegalArgumentException("Invalid locale format: " + str);
-}
-return new Locale(str.substring(0, 2), str.substring(3, 5), str.substring(6));
-}
-}
-}
```

**Ground truth fix:**
```diff
@@ -19,4 +19,7 @@
 }
 char ch3 = str.charAt(3);
+if (ch3 == '_') {
+return new Locale(str.substring(0, 2), "", str.substring(4));
+}
 char ch4 = str.charAt(4);
 if (ch3 < 'A' || ch3 > 'Z' || ch4 < 'A' || ch4 > 'Z') {
```

---

## Math-102

### Patch 1

**Root cause:** The bug is that chi-square goodness-of-fit must rescale expected counts when the total expected count differs from the total observed count, but the buggy method directly used unscaled expected values.

**Why pre-patch overfits:** The pre-patch does address the main statistical root cause by rescaling expected counts, but it does so by modifying the input array, which is not equivalent to the ground-truth fix. That makes it incorrect on reasonable inputs beyond the provided tests, such as reusing the same expected array across multiple chiSquare/chiSquareTest calls or elsewhere in client code, where the array would be unexpectedly rescaled.

**OD refinement (NECESSARY):** The OD refinement changed the rescaling implementation so it no longer mutates the caller-provided expected array in place; instead it creates a scaled copy and uses that locally. This is a real semantic fix, because the pre-patch introduces externally visible side effects: after one call, the input array is altered, so repeated calls or later uses of the same expected array can produce different behavior than intended.

**Verdict:** Because the refinement is NECESSARY and fixes a genuine missing semantic property of the pre-patch (no mutation of inputs), the original patch was overfitting despite passing the tests. It solved the numeric symptom but not correctly enough for general use.

**Before OD (overfitting):**
```diff
@@ -9,4 +9,16 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double expectedSum = 0.0d;
+double observedSum = 0.0d;
+for (int i = 0; i < observed.length; i++) {
+expectedSum += expected[i];
+observedSum += observed[i];
+}
+if (expectedSum != observedSum) {
+double scale = observedSum / expectedSum;
+for (int i = 0; i < expected.length; i++) {
+expected[i] *= scale;
+}
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
```

**After OD (correct):**
```diff
@@ -1,4 +1,7 @@
 public double chiSquare(double[] expected, long[] observed)
 throws IllegalArgumentException {
+if (expected == null || observed == null) {
+throw new IllegalArgumentException("Input arrays cannot be null");
+}
 if ((expected.length < 2) || (expected.length != observed.length)) {
 throw new IllegalArgumentException(
@@ -7,9 +10,26 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
+}
+double expectedSum = 0.0d;
+double observedSum = 0.0d;
+for (int i = 0; i < observed.length; i++) {
+expectedSum += expected[i];
+observedSum += observed[i];
+}
+if (expectedSum != observedSum) {
+double scale = observedSum / expectedSum;
+double[] scaledExpected = new double[expected.length];
+for (int i = 0; i < expected.length; i++) {
+scaledExpected[i] = expected[i] * scale;
+}
+expected = scaledExpected;
 }
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (expected[i] == 0) {
+throw new IllegalArgumentException("Expected value cannot be zero");
+}
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
```

**Ground truth fix:**
```diff
@@ -9,10 +9,27 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0d;
+double sumObserved = 0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+double ratio = 1.0d;
+boolean rescale = false;
+if (Math.abs(sumExpected - sumObserved) > 10E-6) {
+ratio = sumObserved / sumExpected;
+rescale = true;
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (rescale) {
+dev = ((double) observed[i] - ratio * expected[i]);
+sumSq += dev * dev / (ratio * expected[i]);
+} else {
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
 }
+}
 return sumSq;
 }
```

### Patch 2

**Root cause:** The bug is that the chi-square statistic must rescale expected counts when the total expected count differs from the total observed count, instead of always using the raw expected values.

**Why pre-patch overfits:** The pre-patch does address the main statistical root cause by rescaling expected counts before computing the statistic, so it will return the right value for many one-shot calls including the failing tests. However, it is not semantically correct because it mutates the expected input array, violating normal expectations for a pure test-statistic computation and causing incorrect behavior if the same array is reused in subsequent calls or elsewhere.

**OD refinement (NECESSARY):** The OD refinement changed the implementation from mutating the input expected array in place to computing a local scaledExpected value per element, and it also added null/zero-sum guards. The key necessary semantic change is avoiding in-place modification of the caller-provided expected array, because the pre-patch introduces an observable side effect that can corrupt later computations on the same array even though the statistic for a single call is correct.

**Verdict:** Because the refinement introduced a necessary semantic fix?봢liminating the in-place mutation of expected?봳he pre-patch was not genuinely correct. It passed the tests by coincidence of test usage patterns, but would fail on reasonable inputs involving reuse of the expected array.

**Before OD (overfitting):**
```diff
@@ -7,5 +7,15 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
+}
+double sumObserved = 0.0d;
+double sumExpected = 0.0d;
+for (int i = 0; i < observed.length; i++) {
+sumObserved += observed[i];
+sumExpected += expected[i];
+}
+double scale = sumObserved / sumExpected;
+for (int i = 0; i < expected.length; i++) {
+expected[i] *= scale;
 }
 double sumSq = 0.0d;
```

**After OD (correct):**
```diff
@@ -1,4 +1,7 @@
 public double chiSquare(double[] expected, long[] observed)
 throws IllegalArgumentException {
+if (expected == null || observed == null) {
+throw new IllegalArgumentException("Input arrays cannot be null");
+}
 if ((expected.length < 2) || (expected.length != observed.length)) {
 throw new IllegalArgumentException(
@@ -7,11 +10,30 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
 }
+double sumObserved = 0.0d;
+for (long count : observed) {
+sumObserved += count;
+}
+double sumExpected = 0.0d;
+for (double count : expected) {
+sumExpected += count;
+}
+if (sumExpected == 0) {
+throw new IllegalArgumentException("Sum of expected values must be greater than zero");
+}
+double scale = sumObserved / sumExpected;
 double sumSq = 0.0d;
 double dev = 0.0d;
-for (int i = 0; i < observed.length; i++) {
-dev = ((double) observed[i] - expected[i]);
-sumSq += dev * dev / expected[i];
+for (int i = 0; i < expected.length; i++) {
+double scaledExpected = expected[i] * scale;
+if (scaledExpected > 0) { // Avoid division by zero
+dev = ((double) observed[i] - scaledExpected);
+sumSq += dev * dev / scaledExpected;
+} else {
+if (observed[i] != 0) {
+throw new IllegalArgumentException("Scaled expected value cannot be zero when observed value is non-zero");
+}
+}
 }
 return sumSq;
```

**Ground truth fix:**
```diff
@@ -9,10 +9,27 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0d;
+double sumObserved = 0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+double ratio = 1.0d;
+boolean rescale = false;
+if (Math.abs(sumExpected - sumObserved) > 10E-6) {
+ratio = sumObserved / sumExpected;
+rescale = true;
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (rescale) {
+dev = ((double) observed[i] - ratio * expected[i]);
+sumSq += dev * dev / (ratio * expected[i]);
+} else {
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
 }
+}
 return sumSq;
 }
```

### Patch 3

**Root cause:** The bug is that chi-square goodness-of-fit must rescale expected counts when the total expected count differs from the total observed count, but the buggy method always used unscaled expected values.

**Why pre-patch overfits:** The pre-patch does address the main statistical root cause by rescaling expected counts, so it will compute the right statistic for a single isolated call. However, it is not semantically correct as a library method because it modifies the expected array argument, which is an observable side effect not present in the ground-truth fix and can cause incorrect behavior on reasonable repeated-call or array-reuse scenarios outside the provided tests.

**OD refinement (NECESSARY):** The OD refinement changed the implementation from mutating the input expected array in place to computing a per-entry scaledExpected value without modifying the caller's array, and it also added null/zero-sum guards. The key necessary semantic change is avoiding in-place mutation: the pre-patch has a side effect on its input array that can corrupt later uses of the same expected data, whereas the refined version preserves input values while applying the same rescaling logic.

**Verdict:** Because the refinement introduced necessary logic to eliminate the pre-patch's destructive mutation of input data, the two versions are not semantically equivalent. The pre-patch passes the tests but remains overfitting/incorrect due to this missing behavior-preserving logic.

**Before OD (overfitting):**
```diff
@@ -7,5 +7,17 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
+}
+double sumObserved = 0.0d;
+for (long count : observed) {
+sumObserved += count;
+}
+double sumExpected = 0.0d;
+for (double count : expected) {
+sumExpected += count;
+}
+double scale = sumObserved / sumExpected;
+for (int i = 0; i < expected.length; i++) {
+expected[i] *= scale;
 }
 double sumSq = 0.0d;
```

**After OD (correct):**
```diff
@@ -1,4 +1,7 @@
 public double chiSquare(double[] expected, long[] observed)
 throws IllegalArgumentException {
+if (expected == null || observed == null) {
+throw new IllegalArgumentException("Input arrays cannot be null");
+}
 if ((expected.length < 2) || (expected.length != observed.length)) {
 throw new IllegalArgumentException(
@@ -7,11 +10,30 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
 }
+double sumObserved = 0.0d;
+for (long count : observed) {
+sumObserved += count;
+}
+double sumExpected = 0.0d;
+for (double count : expected) {
+sumExpected += count;
+}
+if (sumExpected == 0) {
+throw new IllegalArgumentException("Sum of expected values must be greater than zero");
+}
+double scale = sumObserved / sumExpected;
 double sumSq = 0.0d;
 double dev = 0.0d;
-for (int i = 0; i < observed.length; i++) {
-dev = ((double) observed[i] - expected[i]);
-sumSq += dev * dev / expected[i];
+for (int i = 0; i < expected.length; i++) {
+double scaledExpected = expected[i] * scale;
+if (scaledExpected > 0) { // Avoid division by zero
+dev = ((double) observed[i] - scaledExpected);
+sumSq += dev * dev / scaledExpected;
+} else {
+if (observed[i] != 0) {
+throw new IllegalArgumentException("Scaled expected value cannot be zero when observed value is non-zero");
+}
+}
 }
 return sumSq;
```

**Ground truth fix:**
```diff
@@ -9,10 +9,27 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0d;
+double sumObserved = 0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+double ratio = 1.0d;
+boolean rescale = false;
+if (Math.abs(sumExpected - sumObserved) > 10E-6) {
+ratio = sumObserved / sumExpected;
+rescale = true;
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (rescale) {
+dev = ((double) observed[i] - ratio * expected[i]);
+sumSq += dev * dev / (ratio * expected[i]);
+} else {
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
 }
+}
 return sumSq;
 }
```

### Patch 4

**Root cause:** The chi-square goodness-of-fit statistic must rescale expected counts when the total expected count differs from the total observed count, but the buggy method always used raw expected values.

**Why pre-patch overfits:** The pre-patch does address the main statistical root cause by rescaling expected totals before computing the statistic, so it passes the reported tests for a single call. However, it does so by modifying the caller's expected array in place, which is not part of the intended contract and can break reasonable usages outside the tests, such as reusing the same expected array across multiple chi-square computations or subsequent assertions.

**OD refinement (NECESSARY):** The OD refinement changed the pre-patch from mutating the input expected array in place to computing a per-element scaledExpected value without side effects, and it also made the rescaling unconditional via a scale factor. This semantic change is necessary because the pre-patch corrupts caller-visible input state, so repeated calls or later uses of the same expected array can produce incorrect behavior even though the immediate statistic value is correct for the tested calls.

**Verdict:** Because the refinement introduced a necessary semantic fix?봢liminating destructive mutation of the input array?봳he pre-patch was not genuinely correct. It passed the tests, but only with an implementation that can yield wrong behavior on other valid inputs/usages.

**Before OD (overfitting):**
```diff
@@ -9,4 +9,16 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0.0d;
+double sumObserved = 0.0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+if (sumExpected != sumObserved) {
+double scale = sumObserved / sumExpected;
+for (int i = 0; i < expected.length; i++) {
+expected[i] *= scale;
+}
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
```

**After OD (correct):**
```diff
@@ -1,4 +1,7 @@
 public double chiSquare(double[] expected, long[] observed)
 throws IllegalArgumentException {
+if (expected == null || observed == null) {
+throw new IllegalArgumentException("Input arrays cannot be null");
+}
 if ((expected.length < 2) || (expected.length != observed.length)) {
 throw new IllegalArgumentException(
@@ -7,11 +10,30 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
 }
+double sumObserved = 0.0d;
+for (long count : observed) {
+sumObserved += count;
+}
+double sumExpected = 0.0d;
+for (double count : expected) {
+sumExpected += count;
+}
+if (sumExpected == 0) {
+throw new IllegalArgumentException("Sum of expected values must be greater than zero");
+}
+double scale = sumObserved / sumExpected;
 double sumSq = 0.0d;
 double dev = 0.0d;
-for (int i = 0; i < observed.length; i++) {
-dev = ((double) observed[i] - expected[i]);
-sumSq += dev * dev / expected[i];
+for (int i = 0; i < expected.length; i++) {
+double scaledExpected = expected[i] * scale;
+if (scaledExpected > 0) { // Avoid division by zero
+dev = ((double) observed[i] - scaledExpected);
+sumSq += dev * dev / scaledExpected;
+} else {
+if (observed[i] != 0) {
+throw new IllegalArgumentException("Scaled expected value cannot be zero when observed value is non-zero");
+}
+}
 }
 return sumSq;
```

**Ground truth fix:**
```diff
@@ -9,10 +9,27 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0d;
+double sumObserved = 0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+double ratio = 1.0d;
+boolean rescale = false;
+if (Math.abs(sumExpected - sumObserved) > 10E-6) {
+ratio = sumObserved / sumExpected;
+rescale = true;
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (rescale) {
+dev = ((double) observed[i] - ratio * expected[i]);
+sumSq += dev * dev / (ratio * expected[i]);
+} else {
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
 }
+}
 return sumSq;
 }
```

### Patch 5

**Root cause:** The bug is that chi-square goodness-of-fit must rescale expected counts when the total expected and observed counts differ, but the buggy method always used raw expected values.

**Why pre-patch overfits:** The pre-patch does address the main statistical root cause by rescaling expected counts, but it does so by overwriting the expected array, which is not equivalent to the ground-truth fix and is unsafe API behavior for a pure computation method. On reasonable inputs outside the provided tests, such as reusing the same expected array across multiple chiSquare/chiSquareTest calls, the pre-patch can yield different results because the array has been permanently rescaled.

**OD refinement (NECESSARY):** The OD refinement changed the pre-patch from mutating the input expected array in place to computing a scaledExpected value per element without modifying the caller-provided array, and it also always applies scaling via a local factor. This is semantically important because the pre-patch introduces an observable side effect: after one call, the expected array is altered, so repeated calls or later uses of the same array can produce incorrect behavior, whereas the refined version preserves input data.

**Verdict:** Because the refinement introduced necessary semantic logic?봞voiding mutation of the input array rather than merely refactoring?봳he pre-patch was not genuinely correct. It passes the given tests but remains overfitting due to incorrect side effects on caller-visible state.

**Before OD (overfitting):**
```diff
@@ -7,5 +7,17 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
+}
+double expectedSum = 0.0d;
+double observedSum = 0.0d;
+for (int i = 0; i < observed.length; i++) {
+expectedSum += expected[i];
+observedSum += observed[i];
+}
+if (expectedSum != observedSum) {
+double scale = observedSum / expectedSum;
+for (int i = 0; i < expected.length; i++) {
+expected[i] *= scale;
+}
 }
 double sumSq = 0.0d;
```

**After OD (correct):**
```diff
@@ -1,4 +1,7 @@
 public double chiSquare(double[] expected, long[] observed)
 throws IllegalArgumentException {
+if (expected == null || observed == null) {
+throw new IllegalArgumentException("Input arrays cannot be null");
+}
 if ((expected.length < 2) || (expected.length != observed.length)) {
 throw new IllegalArgumentException(
@@ -7,11 +10,30 @@
 if (!isPositive(expected) || !isNonNegative(observed)) {
 throw new IllegalArgumentException(
-"observed counts must be non-negative and expected counts must be postive");
+"observed counts must be non-negative and expected counts must be positive");
 }
+double sumObserved = 0.0d;
+for (long count : observed) {
+sumObserved += count;
+}
+double sumExpected = 0.0d;
+for (double count : expected) {
+sumExpected += count;
+}
+if (sumExpected == 0) {
+throw new IllegalArgumentException("Sum of expected values must be greater than zero");
+}
+double scale = sumObserved / sumExpected;
 double sumSq = 0.0d;
 double dev = 0.0d;
-for (int i = 0; i < observed.length; i++) {
-dev = ((double) observed[i] - expected[i]);
-sumSq += dev * dev / expected[i];
+for (int i = 0; i < expected.length; i++) {
+double scaledExpected = expected[i] * scale;
+if (scaledExpected > 0) { // Avoid division by zero
+dev = ((double) observed[i] - scaledExpected);
+sumSq += dev * dev / scaledExpected;
+} else {
+if (observed[i] != 0) {
+throw new IllegalArgumentException("Scaled expected value cannot be zero when observed value is non-zero");
+}
+}
 }
 return sumSq;
```

**Ground truth fix:**
```diff
@@ -9,10 +9,27 @@
 "observed counts must be non-negative and expected counts must be postive");
 }
+double sumExpected = 0d;
+double sumObserved = 0d;
+for (int i = 0; i < observed.length; i++) {
+sumExpected += expected[i];
+sumObserved += observed[i];
+}
+double ratio = 1.0d;
+boolean rescale = false;
+if (Math.abs(sumExpected - sumObserved) > 10E-6) {
+ratio = sumObserved / sumExpected;
+rescale = true;
+}
 double sumSq = 0.0d;
 double dev = 0.0d;
 for (int i = 0; i < observed.length; i++) {
+if (rescale) {
+dev = ((double) observed[i] - ratio * expected[i]);
+sumSq += dev * dev / (ratio * expected[i]);
+} else {
 dev = ((double) observed[i] - expected[i]);
 sumSq += dev * dev / expected[i];
 }
+}
 return sumSq;
 }
```

---

## Math-37

### Patch 1

**Root cause:** The bug is missing special-case handling for complex tan/tanh when one component is infinite or when the standard formula overflows, causing undefined Infinity/Infinity or NaN results instead of the mathematically correct limits.

**Why pre-patch overfits:** The pre-patch fixes the tested overflow case for very large finite real values and also happens to handle finite-imaginary, infinite-real inputs, but it does not correctly handle inputs with infinite imaginary part. For example, tanh(1, +Infinity) would compute d as infinite and incorrectly return 1+0i, whereas the correct behavior per the ground truth is NaN, so the patch is overfitting beyond the observed tests.

**OD refinement (NECESSARY):** The refinement added explicit handling for infinite real and infinite imaginary parts, especially returning NaN when the imaginary part is infinite and only returning 짹1+0i when the real part is infinite but the imaginary part is finite. This is semantically important because the pre-patch treated any infinite denominator as a signal to return sign(real)+0i, which conflates valid large-real cases with invalid infinite-imaginary cases.

**Verdict:** Because the refinement introduced necessary missing logic for infinite-component cases that the pre-patch did not implement, the pre-patch is not a fully correct fix. Its broad `Double.isInfinite(d)` fallback is semantically wrong on reasonable inputs outside the failing tests.

**Before OD (overfitting):**
```java
public Complex tanh() {
    if (isNaN) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double d = cosh2a + cos2b;
    if (Double.isInfinite(d)) {
        return createComplex(FastMath.signum(real) * 1.0, 0.0);
    }
    return createComplex(FastMath.sinh(real2) / d,
                         FastMath.sin(imaginary2) / d);
}
```

**After OD (correct):**
```java
public Complex tanh() {
    if (isNaN || Double.isNaN(real) || Double.isNaN(imaginary)) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double sinh2a = FastMath.sinh(real2);
    double sin2b = FastMath.sin(imaginary2);
    double d = cosh2a + cos2b;

    if (Double.isInfinite(real)) {
        if (Double.isInfinite(imaginary)) {
            return NaN;
        } else {
            return createComplex(FastMath.copySign(1.0, real), 0.0);
        }
    } else if (Double.isInfinite(imaginary)) {
        if (FastMath.abs(cos2b) < 1e-10) { // cos(2b) is effectively zero
            return createComplex(0.0, FastMath.copySign(Double.POSITIVE_INFINITY, sin2b));
        }
        return NaN;
    } else if (Double.isInfinite(d)) {
        return createComplex(FastMath.copySign(1.0, sinh2a), 0.0);
    }

    return createComplex(sinh2a / d, sin2b / d);
}
```

### Patch 2

**Root cause:** The bug is missing special-case handling for infinite or very large components in complex tan/tanh, where direct use of sin/cos/sinh/cosh causes indeterminate Infinity/Infinity or NaN instead of the correct limiting values or NaN.

**Why pre-patch overfits:** The pre-patch is not a correct root-cause fix. It patches only tan() by returning 0 짹 i when the denominator overflows, which happens to satisfy the large-imaginary tan tests, but it leaves tanh() unfixed and also gives wrong results for tan() when real is infinite and imaginary is finite (e.g., inf+1i should be NaN, but the patch would return 0+1i).

**OD refinement (NECESSARY):** The OD refinement added explicit semantic handling for infinities in tanh: returning NaN when the imaginary part is infinite, returning 짹1+0i when the real part is infinite, and only using the overflow-based denominator check as a fallback. This is necessary because the pre-patch only changes tan(), and even there it relies on d being infinite rather than correctly distinguishing cases like infinite real parts that should yield NaN.

**Verdict:** Because the refinement introduced necessary missing logic for correct infinity handling rather than a cosmetic rewrite, the pre-patch was overfitting. It passed tests by addressing only one overflow symptom and not the full semantic contract of tan/tanh.

**Before OD (overfitting):**
```java
public Complex tan() {
    if (isNaN) {
        return NaN;
    }

    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosReal2 = FastMath.cos(real2);
    double coshImaginary2 = FastMath.cosh(imaginary2);
    double d = cosReal2 + coshImaginary2;

    if (Double.isInfinite(d)) {
        return createComplex(0, FastMath.signum(imaginary));
    }

    return createComplex(FastMath.sin(real2) / d,
                         FastMath.sinh(imaginary2) / d);
}
```

**After OD (correct):**
```java
public Complex tanh() {
    if (isNaN || Double.isNaN(real) || Double.isNaN(imaginary)) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double sinh2a = FastMath.sinh(real2);
    double sin2b = FastMath.sin(imaginary2);
    double d = cosh2a + cos2b;

    if (Double.isInfinite(real)) {
        if (Double.isInfinite(imaginary)) {
            return NaN;
        } else {
            return createComplex(FastMath.copySign(1.0, real), 0.0);
        }
    } else if (Double.isInfinite(imaginary)) {
        if (FastMath.abs(cos2b) < 1e-10) { // cos(2b) is effectively zero
            return createComplex(0.0, FastMath.copySign(Double.POSITIVE_INFINITY, sin2b));
        }
        return NaN;
    } else if (Double.isInfinite(d)) {
        return createComplex(FastMath.copySign(1.0, sinh2a), 0.0);
    }

    return createComplex(sinh2a / d, sin2b / d);
}
```

### Patch 3

**Root cause:** The bug is missing special handling for overflow/infinite arguments in complex tan/tanh, where direct evaluation of cosh/sinh causes infinities/NaNs instead of the correct limiting values or NaN cases.

**Why pre-patch overfits:** The pre-patch fixes the tested large-imaginary overflow behavior for `tan`, but it does not correctly handle inputs with infinite real part. For example, `tan(??+ i)` would compute an infinite `d` and incorrectly return `(0, 1)` instead of `NaN`, so it is not a complete semantic fix and would fail on reasonable inputs beyond the exercised tests.

**OD refinement (NECESSARY):** The OD refinement added semantic discrimination inside the `Double.isInfinite(d)` case: it returns `(0, sign(imaginary))` only when the denominator is infinite because `cosh(2*imaginary)` overflowed, and otherwise returns `NaN`. This matters because the pre-patch treated every infinite denominator the same, including cases with infinite real part where `tan` should be `NaN`, not `(0, 짹1)`.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic for distinguishing overflow from invalid infinite-real inputs, the pre-patch was overfitting: it passed tests but still produced wrong results for valid edge cases covered by the true fix.

**Before OD (overfitting):**
```java
public Complex tan() {
    if (isNaN) {
        return NaN;
    }

    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosReal2 = FastMath.cos(real2);
    double coshImaginary2 = FastMath.cosh(imaginary2);
    double d = cosReal2 + coshImaginary2;

    if (Double.isInfinite(d)) {
        return createComplex(0, FastMath.signum(imaginary) * 1);
    }

    return createComplex(FastMath.sin(real2) / d,
                         FastMath.sinh(imaginary2) / d);
}
```

**After OD (correct):**
```java
public Complex tan() {
    if (isNaN) {
        return NaN;
    }

    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosReal2 = FastMath.cos(real2);
    double coshImaginary2 = FastMath.cosh(imaginary2);
    double d = cosReal2 + coshImaginary2;

    // Handle cases where the denominator is infinite
    if (Double.isInfinite(d)) {
        if (FastMath.abs(cosReal2) < Double.MAX_VALUE && FastMath.abs(coshImaginary2) == Double.POSITIVE_INFINITY) {
            return createComplex(0, FastMath.signum(imaginary) * 1);
        } else {
            return NaN;
        }
    }

    // Handle cases where the real part is 짹?/2
    if (FastMath.abs(real) == FastMath.PI / 2 && imaginary == 0) {
        return createComplex(FastMath.signum(real) * Double.POSITIVE_INFINITY, Double.NaN);
    }

    return createComplex(FastMath.sin(real2) / d,
                         FastMath.sinh(imaginary2) / d);
}
```

### Patch 4

**Root cause:** The bug is that Complex.tan/tanh use formulas that overflow or yield indeterminate Infinity/Infinity and NaN for large or infinite components, instead of returning the correct limiting values or NaN for the appropriate infinite cases.

**Why pre-patch overfits:** The pre-patch fixes some large-real/infinite-real cases for tanh by mapping an infinite denominator to 짹1 + 0i, so it passes the observed tests, but it is not a complete semantic fix. For example, with tanh(1 + Infinity i), cos(2b) and sin(2b) become NaN, so d is NaN rather than infinite, and the pre-patch returns NaN only accidentally; more importantly, it lacks the explicit infinite-imaginary logic required by the true fix and does not address the companion tan() root-cause behavior at all.

**OD refinement (NECESSARY):** The OD refinement added explicit handling for infinite real and imaginary parts, distinguishing cases like infinite real with finite imaginary (which should return 짹1 + 0i) from infinite imaginary cases (which should return NaN in tanh). The pre-patch only checked whether the denominator d became infinite and then returned sign(real), which is an indirect heuristic and does not correctly encode the semantic rules for all infinite-input combinations.

**Verdict:** Because the refinement change is NECESSARY and adds missing semantic handling for infinite arguments rather than mere restructuring, the pre-patch was an overfitting partial fix that passed tests without fully implementing the correct behavior.

**Before OD (overfitting):**
```java
public Complex tanh() {
    if (isNaN) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double sinh2a = FastMath.sinh(real2);
    double sin2b = FastMath.sin(imaginary2);
    double d = cosh2a + cos2b;
    if (Double.isInfinite(d)) {
        return createComplex(FastMath.signum(real), 0);
    }
    return createComplex(sinh2a / d,
                         sin2b / d);
}
```

**After OD (correct):**
```java
public Complex tanh() {
    if (isNaN || Double.isNaN(real) || Double.isNaN(imaginary)) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double sinh2a = FastMath.sinh(real2);
    double sin2b = FastMath.sin(imaginary2);
    double d = cosh2a + cos2b;

    if (Double.isInfinite(real)) {
        if (Double.isInfinite(imaginary)) {
            return NaN;
        } else {
            return createComplex(FastMath.copySign(1.0, real), 0.0);
        }
    } else if (Double.isInfinite(imaginary)) {
        if (FastMath.abs(cos2b) < 1e-10) { // cos(2b) is effectively zero
            return createComplex(0.0, FastMath.copySign(Double.POSITIVE_INFINITY, sin2b));
        }
        return NaN;
    } else if (Double.isInfinite(d)) {
        return createComplex(FastMath.copySign(1.0, sinh2a), 0.0);
    }

    return createComplex(sinh2a / d, sin2b / d);
}
```

### Patch 5

**Root cause:** The bug is that Complex.tanh() (and similarly tan()) does not explicitly handle infinite components, so the standard formula produces indeterminate Infinity/Infinity or invalid trig-of-infinity results and returns NaN where the mathematically correct limit is defined.

**Why pre-patch overfits:** The pre-patch fixes the tested large-|real| overflow behavior and the specific infOne/negInfOne cases by returning 짹1 when d becomes infinite, but it does not correctly handle all infinite-input combinations. In particular, for finite real with infinite imaginary, cos(imaginary2) is NaN so the result remains NaN, but for real = 짹??and imaginary = 짹?? d is infinite and the pre-patch returns 짹1 + 0i instead of NaN, so it is semantically incomplete beyond the observed tests.

**OD refinement (NECESSARY):** The refinement added explicit case analysis for infinite real and infinite imaginary parts instead of only checking whether the denominator d is infinite. This is necessary because the pre-patch incorrectly maps any infinite d to (sign(real), 0), which is wrong when imaginary is infinite: for example tanh(1 + i?? should be NaN per the intended semantics, but the pre-patch would return (1, 0).

**Verdict:** Because the OD refinement introduced necessary missing logic for infinite-imaginary and both-infinite cases, the pre-patch was not a fully correct fix. It passed tests by handling only a subset of the root-cause scenarios, so it is a true overfitting patch.

**Before OD (overfitting):**
```java
public Complex tanh() {
    if (isNaN) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double d = cosh2a + cos2b;
    if (Double.isInfinite(d)) {
        return createComplex(FastMath.copySign(1.0, real), 0.0);
    }
    return createComplex(FastMath.sinh(real2) / d,
                         FastMath.sin(imaginary2) / d);
}
```

**After OD (correct):**
```java
public Complex tanh() {
    if (isNaN || Double.isNaN(real) || Double.isNaN(imaginary)) {
        return NaN;
    }
    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosh2a = FastMath.cosh(real2);
    double cos2b = FastMath.cos(imaginary2);
    double sinh2a = FastMath.sinh(real2);
    double sin2b = FastMath.sin(imaginary2);
    double d = cosh2a + cos2b;

    if (Double.isInfinite(real)) {
        if (Double.isInfinite(imaginary)) {
            return NaN;
        } else {
            return createComplex(FastMath.copySign(1.0, real), 0.0);
        }
    } else if (Double.isInfinite(imaginary)) {
        if (FastMath.abs(cos2b) < 1e-10) { // cos(2b) is effectively zero
            return createComplex(0.0, FastMath.copySign(Double.POSITIVE_INFINITY, sin2b));
        }
        return NaN;
    } else if (Double.isInfinite(d)) {
        return createComplex(FastMath.copySign(1.0, sinh2a), 0.0);
    }

    return createComplex(sinh2a / d, sin2b / d);
}
```

### Patch 6

**Root cause:** The bug is missing special handling for tan/tanh when one component is infinite or so large that cosh/sinh overflow, causing indeterminate Infinity/Infinity divisions and NaN instead of the correct limiting values.

**Why pre-patch overfits:** The pre-patch only fixes the large-imaginary overflow case for tan by checking whether d is infinite, which is enough for the shown failing tests, but it does not correctly handle infinite real inputs. For example, tan(inf + i) yields d = inf and signum(imaginary) = 0, so the pre-patch returns (0, 0) instead of NaN; thus it is not a semantically correct general fix.

**OD refinement (NECESSARY):** The OD refinement changed the infinite-denominator handling from always returning (0, sign(imaginary)) to distinguishing whether overflow came from the imaginary-side hyperbolic term or the real-side trigonometric term. This matters because for inputs with infinite real part, the correct result is NaN (per the ground truth), whereas the pre-patch would incorrectly return a finite value such as (0, 0).

**Verdict:** Because the refinement added necessary semantic logic to separate overflow causes, the pre-patch was overfitting to the tested large-imaginary cases. It passed tests but still produced wrong results on reasonable untested inputs such as infinite real parts.

**Before OD (overfitting):**
```java
public Complex tan() {
    if (isNaN) {
        return NaN;
    }

    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double d = FastMath.cos(real2) + FastMath.cosh(imaginary2);

    if (Double.isInfinite(d)) {
        return createComplex(0, FastMath.signum(imaginary));
    }

    return createComplex(FastMath.sin(real2) / d,
                         FastMath.sinh(imaginary2) / d);
}
```

**After OD (correct):**
```java
public Complex tan() {
    if (isNaN) {
        return NaN;
    }

    double real2 = 2.0 * real;
    double imaginary2 = 2.0 * imaginary;
    double cosReal2 = FastMath.cos(real2);
    double coshImaginary2 = FastMath.cosh(imaginary2);
    double d = cosReal2 + coshImaginary2;

    if (Double.isInfinite(d)) {
        // Handle the case where the denominator is infinite due to large values
        if (FastMath.abs(cosReal2) == Double.POSITIVE_INFINITY) {
            return createComplex(FastMath.signum(FastMath.sin(real2)), 0);
        } else if (FastMath.abs(coshImaginary2) == Double.POSITIVE_INFINITY) {
            return createComplex(0, FastMath.signum(FastMath.sinh(imaginary2)));
        }
    }

    return createComplex(FastMath.sin(real2) / d,
                         FastMath.sinh(imaginary2) / d);
}
```

---

## Math-39

**Root cause:** The integrator can use an initial step size larger than the remaining integration interval, causing intermediate Runge-Kutta stage derivative evaluations to occur past the target time.

**Why pre-patch overfits:** The pre-patch addresses the failing test's forward case, but it does not correctly handle the same root cause for backward integration. On reasonable inputs with `t < stepStart`, it can replace a properly signed initialized step with an incorrect value due to using the raw signed interval length instead of its magnitude, so it is not a general correct fix.

**OD refinement (NECESSARY):** The refinement changed the cap on the initial step from `Math.min(hNew, t - stepStart)` to `Math.min(hNew, Math.abs(t - stepStart))`. This is semantically important because the pre-patch only works for forward integration; in backward integration `t - stepStart` is negative, so it can force an invalid negative cap and break the sign of the initialized step.

**Verdict:** Because the refinement is NECESSARY and fixes a real semantic bug in the pre-patch's step limiting logic, the original patch was overfitting to the tested forward-integration scenario rather than providing a fully correct repair.

**Before OD (overfitting):**
```diff
@@ -52,6 +52,6 @@
 }
 }
-hNew = initializeStep(forward, getOrder(), scale,
-stepStart, y, yDotK[0], yTmp, yDotK[1]);
+hNew = Math.min(initializeStep(forward, getOrder(), scale,
+stepStart, y, yDotK[0], yTmp, yDotK[1]), Math.abs(t - stepStart));
 firstTime = false;
 }
```

**After OD (correct):**
```diff
@@ -54,4 +54,6 @@
 hNew = initializeStep(forward, getOrder(), scale,
 stepStart, y, yDotK[0], yTmp, yDotK[1]);
+// Ensure the initial step size does not exceed the remaining interval
+hNew = Math.min(hNew, Math.abs(t - stepStart));
 firstTime = false;
 }
```

**Ground truth fix:**
```diff
@@ -58,4 +58,13 @@
 
 stepSize = hNew;
+if (forward) {
+if (stepStart + stepSize >= t) {
+stepSize = t - stepStart;
+}
+} else {
+if (stepStart + stepSize <= t) {
+stepSize = t - stepStart;
+}
+}
 
 // next stages
```

---

## Math-97

### Patch 1

**Root cause:** The bug is that solve(min, max) treats sign == 0 as a normal bracketing case instead of directly returning the endpoint that is already an exact root.

**Why pre-patch overfits:** The pre-patch only checks near-zero endpoints when `sign > 0`, which cannot happen when one endpoint is exactly zero because then `sign == 0`. As a result, exact endpoint roots are not handled by the intended endpoint logic and are instead delegated to the general solver, so the patch does not correctly implement the root-cause fix beyond the specific tests it happened to pass.

**OD refinement (NECESSARY):** The refinement added an explicit `else if (sign == 0)` branch that returns `min` or `max` when one endpoint evaluates exactly to zero. This is semantically important because in the pre-patch, `sign == 0` falls into the solver call path, which is not equivalent to immediately recognizing an endpoint root and can yield incorrect behavior or state handling on exact-endpoint-root inputs.

**Verdict:** Because the OD refinement introduced necessary missing logic for the `sign == 0` case, the pre-patch was not semantically complete. Passing tests was incidental; the refined patch fixed a real gap rather than making a cosmetic or equivalent change.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,3 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
-
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 clearResult();
 verifyInterval(min, max);
@@ -12,6 +10,11 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) < getAbsoluteAccuracy()) {
+return min;
+} else if (Math.abs(yMax) < getAbsoluteAccuracy()) {
+return max;
+}
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
```

**After OD (correct):**
```diff
@@ -1,5 +1,3 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
-
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 clearResult();
 verifyInterval(min, max);
@@ -12,6 +10,11 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) < getAbsoluteAccuracy()) {
+return min;
+} else if (Math.abs(yMax) < getAbsoluteAccuracy()) {
+return max;
+}
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -19,4 +22,7 @@
 "  Endpoints: [" + min + "," + max + "]" +
 "  Values: [" + yMin + "," + yMax + "]");
+} else if (sign == 0) {
+// one of the endpoints is exactly zero
+return (yMin == 0) ? min : max;
 } else {
 // solve using only the first endpoint as initial guess
```

**Ground truth fix:**
```diff
@@ -12,6 +12,13 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) <= functionValueAccuracy) {
+setResult(min, 0);
+ret = min;
+} else if (Math.abs(yMax) <= functionValueAccuracy) {
+setResult(max, 0);
+ret = max;
+} else {
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -19,8 +26,15 @@
 "  Endpoints: [" + min + "," + max + "]" +
 "  Values: [" + yMin + "," + yMax + "]");
-} else {
+}
+} else if (sign < 0){
 // solve using only the first endpoint as initial guess
 ret = solve(min, yMin, max, yMax, min, yMin);
+} else {
 // either min or max is a root
+if (yMin == 0.0) {
+ret = min;
+} else {
+ret = max;
+}
 }
 
```

### Patch 2

**Root cause:** The bug is that the solver treats endpoint roots incorrectly by branching on yMin * yMax without first handling the case where one endpoint value is zero or sufficiently close to zero.

**Why pre-patch overfits:** The pre-patch fixes the provided tests only because for sin(pi) the floating-point value is a tiny nonzero number, making sign < 0 and allowing the internal solver to still converge. But if an endpoint is exactly zero for a reasonable function/input, sign == 0 falls into the else branch and calls solve(...) instead of immediately returning the endpoint root, so it does not correctly implement the root-cause fix.

**OD refinement (NECESSARY):** The refinement moves endpoint-root detection ahead of the sign test and checks both endpoints against a tolerance using <=, so cases with sign == 0 are returned directly instead of being sent into the internal solve routine. This is semantically important because the pre-patch only checks near-zero endpoints when sign > 0, leaving exact endpoint roots (sign == 0) to the wrong branch.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch was missing required logic for the sign == 0 endpoint-root case. It passed the observed tests by coincidence of floating-point behavior rather than by a fully correct fix.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,3 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
-
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 clearResult();
 verifyInterval(min, max);
@@ -12,6 +10,11 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) < getAbsoluteAccuracy()) {
+return min;
+} else if (Math.abs(yMax) < getAbsoluteAccuracy()) {
+return max;
+} else {
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -19,4 +22,5 @@
 "  Endpoints: [" + min + "," + max + "]" +
 "  Values: [" + yMin + "," + yMax + "]");
+}
 } else {
 // solve using only the first endpoint as initial guess
```

**After OD (correct):**
```diff
@@ -1,5 +1,3 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
-
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 clearResult();
 verifyInterval(min, max);
@@ -10,8 +8,15 @@
 double yMax = f.value(max);
 
-// Verify bracketing
+// Define a tolerance level
+double tolerance = getAbsoluteAccuracy();
+
+// Verify bracketing with tolerance
+if (Math.abs(yMin) <= tolerance) {
+return min;
+} else if (Math.abs(yMax) <= tolerance) {
+return max;
+} else {
 double sign = yMin * yMax;
 if (sign >= 0) {
-// check if either value is close to a zero
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -24,4 +29,5 @@
 // either min or max is a root
 }
+}
 
 return ret;
```

**Ground truth fix:**
```diff
@@ -12,6 +12,13 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) <= functionValueAccuracy) {
+setResult(min, 0);
+ret = min;
+} else if (Math.abs(yMax) <= functionValueAccuracy) {
+setResult(max, 0);
+ret = max;
+} else {
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -19,8 +26,15 @@
 "  Endpoints: [" + min + "," + max + "]" +
 "  Values: [" + yMin + "," + yMax + "]");
-} else {
+}
+} else if (sign < 0){
 // solve using only the first endpoint as initial guess
 ret = solve(min, yMin, max, yMax, min, yMin);
+} else {
 // either min or max is a root
+if (yMin == 0.0) {
+ret = min;
+} else {
+ret = max;
+}
 }
 
```

### Patch 3

**Root cause:** The bug is that the solver treats endpoint roots as non-bracketing cases and throws or recurses instead of explicitly returning an endpoint when yMin * yMax == 0.

**Why pre-patch overfits:** The pre-patch fixes the specific failing tests only because those endpoint values are close enough to zero to avoid the bad path, but it does not correctly implement the endpoint-root case in general. For reasonable inputs where an endpoint evaluates to exactly zero (or sign == 0 without entering the sign > 0 branch), it incorrectly calls the internal solver instead of immediately returning the endpoint, so it does not fully fix the root cause.

**OD refinement (NECESSARY):** The refinement changes the condition from handling only sign > 0 to handling sign >= 0, and adds explicit checks for yMin == 0 and yMax == 0 before any near-zero tolerance logic. This is necessary because in the pre-patch, the sign == 0 case falls into the recursive solve branch, which is semantically wrong for exact endpoint roots and can mis-handle cases where an endpoint is exactly zero but not caught by the sign > 0 branch.

**Verdict:** Because the refinement added necessary missing logic for the sign == 0 case, the pre-patch was not semantically complete. It passed the observed tests by relying on tolerance-based coincidence rather than correctly handling exact endpoint roots.

**Before OD (overfitting):**
```diff
@@ -1,22 +1,14 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
-
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 clearResult();
 verifyInterval(min, max);
-
 double ret = Double.NaN;
-
 double yMin = f.value(min);
 double yMax = f.value(max);
-
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
 // neither value is close to zero and min and max do not bracket root.
-throw new IllegalArgumentException
-("Function values at endpoints do not have different signs." +
-"  Endpoints: [" + min + "," + max + "]" +
-"  Values: [" + yMin + "," + yMax + "]");
+throw new IllegalArgumentException("Function values at endpoints do not have different signs." + "  Endpoints: [" + min + "," + max + "]" + "  Values: [" + yMin + "," + yMax + "]");
 } else {
 // solve using only the first endpoint as initial guess
@@ -24,5 +16,4 @@
 // either min or max is a root
 }
-
 return ret;
 }
```

**After OD (correct):**
```diff
@@ -1,4 +1,3 @@
-public double solve(double min, double max) throws MaxIterationsExceededException,
-FunctionEvaluationException {
+public double solve(double min, double max) throws MaxIterationsExceededException, FunctionEvaluationException {
 
 clearResult();
@@ -13,6 +12,19 @@
 double sign = yMin * yMax;
 if (sign >= 0) {
-// check if either value is close to a zero
-// neither value is close to zero and min and max do not bracket root.
+// Check if either value is exactly zero
+if (yMin == 0) {
+return min;
+}
+if (yMax == 0) {
+return max;
+}
+// Check if either value is close to zero
+if (Math.abs(yMin) < getAbsoluteAccuracy()) {
+return min;
+}
+if (Math.abs(yMax) < getAbsoluteAccuracy()) {
+return max;
+}
+// Neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
 ("Function values at endpoints do not have different signs." +
@@ -20,7 +32,7 @@
 "  Values: [" + yMin + "," + yMax + "]");
 } else {
-// solve using only the first endpoint as initial guess
+// Solve using only the first endpoint as initial guess
 ret = solve(min, yMin, max, yMax, min, yMin);
-// either min or max is a root
+// Either min or max is a root
 }
 
```

**Ground truth fix:**
```diff
@@ -12,6 +12,13 @@
 // Verify bracketing
 double sign = yMin * yMax;
-if (sign >= 0) {
+if (sign > 0) {
 // check if either value is close to a zero
+if (Math.abs(yMin) <= functionValueAccuracy) {
+setResult(min, 0);
+ret = min;
+} else if (Math.abs(yMax) <= functionValueAccuracy) {
+setResult(max, 0);
+ret = max;
+} else {
 // neither value is close to zero and min and max do not bracket root.
 throw new IllegalArgumentException
@@ -19,8 +26,15 @@
 "  Endpoints: [" + min + "," + max + "]" +
 "  Values: [" + yMin + "," + yMax + "]");
-} else {
+}
+} else if (sign < 0){
 // solve using only the first endpoint as initial guess
 ret = solve(min, yMin, max, yMax, min, yMin);
+} else {
 // either min or max is a root
+if (yMin == 0.0) {
+ret = min;
+} else {
+ret = max;
+}
 }
 
```

---

## Mockito-5

### Patch 1

**Root cause:** VerificationOverTimeImpl had a direct catch clause for a JUnit-specific exception type, which introduced an unwanted runtime dependency on JUnit and was too narrow because verification failures can also surface as other AssertionError subclasses.

**Why pre-patch overfits:** The pre-patch fixes the classloading dependency issue by removing the explicit JUnit exception reference, but it does not fully preserve the method's semantics. If delegate.verify(data) throws an AssertionError subtype other than MockitoAssertionError, the pre-patch will now escape immediately instead of being processed by handleVerifyException and retried until timeout, so it is incorrect on reasonable inputs beyond the triggering test.

**OD refinement (NECESSARY):** The OD refinement changed the handler from catching only MockitoAssertionError to catching all AssertionError, thereby preserving the intended retry/timeout behavior for non-Mockito assertion failures after removing the JUnit-specific catch. This is necessary because the pre-patch simply deleted the JUnit-specific catch, leaving ordinary AssertionError-based verification failures unhandled.

**Verdict:** Because the refinement added missing, behaviorally important logic (classified as NECESSARY), the pre-patch was an overfitting fix: it removed the JUnit dependency but failed to correctly handle the broader AssertionError cases that the ground-truth fix covers.

**Before OD (overfitting):**
```diff
@@ -15,7 +15,4 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
-error = handleVerifyException(e);
-}
 }
 
```

**After OD (correct):**
```diff
@@ -12,8 +12,5 @@
 error = null;
 }
-} catch (MockitoAssertionError e) {
-error = handleVerifyException(e);
-}
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+} catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

**Ground truth fix:**
```diff
@@ -15,5 +15,5 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

### Patch 2

**Root cause:** The bug is that verify() explicitly catches a JUnit-specific assertion exception type, creating an unwanted runtime dependency on JUnit instead of handling assertion failures framework-independently.

**Why pre-patch overfits:** The pre-patch fixes the classloading dependency exposed by the failing test, but it does not fully fix the intended behavior of verification-over-time. If delegate.verify(data) throws an AssertionError that is not a MockitoAssertionError, the pre-patch will no longer retry/handle it correctly, so it can fail on reasonable inputs outside the given test.

**OD refinement (NECESSARY):** The refinement replaced the narrow MockitoAssertionError-only handling in the pre-patch with a catch of AssertionError, thereby restoring handling of non-Mockito assertion failures without referencing JUnit classes. This is necessary because simply deleting the JUnit-specific catch removes the dependency but also changes behavior by letting other assertion failures escape immediately instead of being processed by handleVerifyException during over-time verification.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch was missing required logic rather than merely being refactored. It removed the JUnit dependency but did not preserve the generic assertion-handling semantics that the correct fix restores.

**Before OD (overfitting):**
```diff
@@ -15,7 +15,5 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
-error = handleVerifyException(e);
-}
+// Removed JUnit-specific exception handling
 }
 
```

**After OD (correct):**
```diff
@@ -12,8 +12,5 @@
 error = null;
 }
-} catch (MockitoAssertionError e) {
-error = handleVerifyException(e);
-}
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+} catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

**Ground truth fix:**
```diff
@@ -15,5 +15,5 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

### Patch 3

**Root cause:** The bug is that VerificationOverTimeImpl directly references a JUnit-specific exception type in a catch clause, creating an unwanted runtime/class-loading dependency on JUnit instead of handling assertion failures generically.

**Why pre-patch overfits:** The pre-patch fixes the class-loading dependency exposed by the failing test, but it does not fully fix the intended behavior of the method. On reasonable inputs where delegate.verify(data) throws an AssertionError that is not a MockitoAssertionError, the pre-patch would now fail immediately rather than continue polling and recording the error as designed.

**OD refinement (NECESSARY):** The refinement changed the exception handling from catching only MockitoAssertionError to catching all AssertionError, thereby preserving the retry/timeout verification behavior for non-Mockito assertion failures while still removing the JUnit-specific dependency. This is necessary because the pre-patch merely deletes the JUnit-specific catch, which changes semantics by letting other AssertionError subclasses escape immediately instead of being processed by handleVerifyException.

**Verdict:** Because the refinement introduced necessary missing logic?봥eneric AssertionError handling?봳he pre-patch was not semantically equivalent to the correct fix. It passed tests by removing the dependency, but overfit by dropping required verification behavior for other assertion failures.

**Before OD (overfitting):**
```diff
@@ -15,7 +15,5 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
-error = handleVerifyException(e);
-}
+// Removed JUnit specific exception handling
 }
 
```

**After OD (correct):**
```diff
@@ -12,8 +12,5 @@
 error = null;
 }
-} catch (MockitoAssertionError e) {
-error = handleVerifyException(e);
-}
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+} catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

**Ground truth fix:**
```diff
@@ -15,5 +15,5 @@
 error = handleVerifyException(e);
 }
-catch (org.mockito.exceptions.verification.junit.ArgumentsAreDifferent e) {
+catch (AssertionError e) {
 error = handleVerifyException(e);
 }
```

---

