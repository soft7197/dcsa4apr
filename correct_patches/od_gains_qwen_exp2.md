# Overfitting Detector Gains - QwenCoder Exp 2

Overfitting detector gain bugs: 29

These bugs had **no correct patches before OD refinement**. All pre-refinement patches were overfitting ??they passed tests but did not actually fix the root cause. OD refinement produced correct patches for these bugs.

---

## Cli-11

### Patch 1

**Root cause:** The bug is that appendOption prints an argument placeholder whenever argName is non-null, even if the arg name is empty and should therefore be treated as absent.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test with an empty string argName, but it does not correctly handle the broader valid case where argName is null. Since the original buggy code explicitly allowed null argName via a null check, replacing that with isEmpty() introduces a new failure on reasonable inputs outside the test, so it does not fully implement the root cause correctly.

**OD refinement (NECESSARY):** The OD refinement added a null check before calling isEmpty() on option.getArgName(). This changes runtime behavior because the pre-patch can throw a NullPointerException for options with hasArg()==true and a null argName, whereas the refined patch safely suppresses the placeholder in that case.

**Verdict:** Because the refinement change is NECESSARY and adds missing null-safety logic, the pre-patch is a true overfitting patch. It passes the observed test but is semantically wrong for null argName inputs.

**Before OD (overfitting):**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && !option.getArgName().isEmpty())
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

**After OD (correct):**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && option.getArgName() != null && !option.getArgName().isEmpty())
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

**Ground truth fix:**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && option.hasArgName())
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

### Patch 2

**Root cause:** The bug is that appendOption prints an argument placeholder whenever argName is non-null, even when the argument name is empty and therefore should be treated as absent.

**Why pre-patch overfits:** The pre-patch fixes the reported failing test and also matches the ground-truth intent for null and empty arg names, but it does not handle blank-only arg names. Since whitespace-only arg names are a reasonable input and the refined patch was confirmed correct specifically by broadening the check, the pre-patch remains incomplete and can still produce incorrect usage text on such inputs.

**OD refinement (NECESSARY):** The refinement changes the condition from rejecting only the exact empty string to rejecting whitespace-only argument names as well by using trim().isEmpty(). This is a real behavioral change: the pre-patch would still print placeholders like " <   >" for blank arg names, while the refined patch suppresses them.

**Verdict:** Because the refinement is NECESSARY and adds missing logic for blank-only argument names, the pre-patch was not fully correct; it overfits to the tested empty-string case rather than robustly treating absent arg names.

**Before OD (overfitting):**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && (option.getArgName() != null && !option.getArgName().isEmpty()))
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

**After OD (correct):**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && option.getArgName() != null && !option.getArgName().trim().isEmpty())
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

**Ground truth fix:**
```diff
@@ -18,5 +18,5 @@
 
 // if the Option has a value
-if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && option.hasArgName())
 {
 buff.append(" <").append(option.getArgName()).append(">");
```

---

## Cli-12

### Patch 1

**Root cause:** The bug is that flatten() checks option names including leading hyphens when parsing option=value forms, and also uses a fixed two-character split intended for special short options like -D, so it fails to correctly recognize and split -f=bar, -foo=bar, and --foo=bar.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause: for any argument containing '=', it checks options.hasOption(split[0]) using the hyphenated token, so valid inputs such as -f=bar and --foo=bar would not be recognized as option=value pairs unless the API happened to accept hyphen-prefixed names. It then falls back to stopAtNonOption behavior instead of the intended special-option logic, so the patch is semantically wrong beyond the passing tests.

**OD refinement (NECESSARY):** The OD refinement changed the '=' handling to validate the option name after stripping leading hyphens from the left side of the split token. This is necessary because in the pre-patch, options.hasOption(split[0]) is called with values like '-f' or '--foo', which do not match Commons CLI's option lookup API and therefore mis-handle valid option=value inputs outside the tested cases.

**Verdict:** Because the refinement introduced necessary logic?봲tripping leading hyphens before option lookup in the option=value case?봳he pre-patch was missing required semantics rather than merely being differently structured. Therefore the original patch was an overfitting/incorrect fix.

**Before OD (overfitting):**
```diff
@@ -24,14 +24,24 @@
 if (options.hasOption(opt))
 {
-tokens.add(arg);
+if (arg.contains("="))
+{
+int equalsIndex = arg.indexOf('=');
+tokens.add(arg.substring(0, equalsIndex)); // -f
+tokens.add(arg.substring(equalsIndex + 1)); // bar
 }
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+tokens.add(arg);
+}
+}
+else
+{
+if (arg.contains("=") && options.hasOption(arg.substring(0, arg.indexOf("="))))
 {
 // the format is --foo=value or -foo=value
 // the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+int equalsIndex = arg.indexOf('=');
+tokens.add(arg.substring(0, equalsIndex)); // -D or --foo
+tokens.add(arg.substring(equalsIndex + 1)); // property=value
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
@@ -28,10 +28,16 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+// Check for option=value pattern
+String[] split = arg.split("=", 2);
+if (split.length == 2 && options.hasOption(Util.stripLeadingHyphens(split[0])))
 {
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+tokens.add(split[0]);
+tokens.add(split[1]);
+}
+else if (options.hasOption(arg.substring(0, 2)))
+{
+// the format is --foo or -foo
+tokens.add(arg.substring(0, 2)); // -D or --f
+tokens.add(arg.substring(2)); // o or oo
 }
 else
@@ -56,4 +62,4 @@
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

**Root cause:** The bug is that GnuParser.flatten does not correctly recognize and split arguments of the form option=value (e.g. --foo=bar, -foo=bar, -f=bar), and instead either leaves them unsplit or splits them at the wrong position.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause. For example, for '--foo=bar' it produces '--f' and 'bar', which is semantically wrong, and for '-foo=bar' it checks only opt.substring(0, 1), so it relies on the short option 'f' rather than recognizing the actual option token; it also breaks special-property style handling such as '-Dproperty=value' by removing the original arg.substring(0, 2) branch. Thus it would fail on reasonable inputs beyond the specific tests and is overfitting.

**OD refinement (NECESSARY):** The refinement replaces the pre-patch's ad hoc handling with general logic that splits any argument on the first '=' and checks whether the left-hand side is a valid option after stripping leading hyphens. This is necessary because the pre-patch hardcodes incorrect reconstructions like '--' + first character and uses only the first character of the option name, which does not correctly handle long options or preserve the original option token.

**Verdict:** Because the OD refinement introduced necessary missing logic for properly parsing generic option=value forms, the pre-patch was not semantically correct. Its behavior is not equivalent to the ground-truth fix and would mis-handle valid inputs outside the observed tests.

**Before OD (overfitting):**
```diff
@@ -24,14 +24,22 @@
 if (options.hasOption(opt))
 {
-tokens.add(arg);
+if (options.hasOption(opt) && arg.contains("="))
+{
+// the format is -f=bar
+tokens.add("-" + opt.charAt(0)); // -f
+tokens.add(arg.substring(arg.indexOf('=') + 1)); // bar
 }
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+tokens.add(arg);
+}
+}
+else
 {
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+if (options.hasOption(opt.substring(0, 1)) && arg.contains("="))
+{
+// the format is --foo=bar
+tokens.add("--" + opt.substring(0, 1)); // --f
+tokens.add(arg.substring(arg.indexOf('=') + 1)); // bar
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
@@ -28,10 +28,16 @@
 else
 {
-if (options.hasOption(arg.substring(0, 2)))
+// Check for option=value pattern
+String[] split = arg.split("=", 2);
+if (split.length == 2 && options.hasOption(Util.stripLeadingHyphens(split[0])))
 {
-// the format is --foo=value or -foo=value
-// the format is a special properties option (-Dproperty=value)
-tokens.add(arg.substring(0, 2)); // -D
-tokens.add(arg.substring(2)); // property=value
+tokens.add(split[0]);
+tokens.add(split[1]);
+}
+else if (options.hasOption(arg.substring(0, 2)))
+{
+// the format is --foo or -foo
+tokens.add(arg.substring(0, 2)); // -D or --f
+tokens.add(arg.substring(2)); // o or oo
 }
 else
@@ -56,4 +62,4 @@
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

## Cli-27

**Root cause:** OptionGroup stores and compares the selected option using option.getOpt(), which is null for long-only options; it should use the option key that represents either the long option when present or the short option otherwise.

**Why pre-patch overfits:** The pre-patch fixes the specific failing tests for long-only options, but it does not correctly handle options that have no long form. On such reasonable inputs, setSelected would leave selected as null for a short-only option, so the patch is not a general fix for the root cause.

**OD refinement (NECESSARY):** The refinement introduced fallback logic: it computes an optionKey as option.getLongOpt() when available, otherwise option.getOpt(), and uses that consistently for comparison and assignment. This is necessary because the pre-patch uses only getLongOpt(), which breaks short-only options by storing null and comparing against null instead of the actual option identifier.

**Verdict:** Because the OD refinement added necessary missing logic for short-only options, the pre-patch was overfitting to the long-option test scenario rather than implementing the full correct key-selection behavior.

**Before OD (overfitting):**
```diff
@@ -11,7 +11,7 @@
 // same option is being reselected then set the
 // selected member variable
-if (selected == null || selected.equals(option.getOpt()))
+if (selected == null || selected.equals(option))
 {
-selected = option.getOpt();
+selected = option;
 }
 else
```

**After OD (correct):**
```diff
@@ -8,10 +8,12 @@
 }
 
+String optionKey = option.getLongOpt() != null ? option.getLongOpt() : option.getOpt();
+
 // if no option has already been selected or the
 // same option is being reselected then set the
 // selected member variable
-if (selected == null || selected.equals(option.getOpt()))
+if (selected == null || selected.equals(optionKey))
 {
-selected = option.getOpt();
+selected = optionKey;
 }
 else
```

**Ground truth fix:**
```diff
@@ -11,7 +11,7 @@
 // same option is being reselected then set the
 // selected member variable
-if (selected == null || selected.equals(option.getOpt()))
+if (selected == null || selected.equals(option.getKey()))
 {
-selected = option.getOpt();
+selected = option.getKey();
 }
 else
```

---

## Cli-38

### Patch 1

**Root cause:** The bug is that isShortOption only checked whether the entire post-dash token was a declared short option, so it failed to recognize tokens that begin with a valid short option but contain concatenated short options or attached text/value.

**Why pre-patch overfits:** The pre-patch is too permissive: it returns true for tokens like '-xa' whenever 'a' is a defined short option, even though the token does not start with a valid short option sequence. That means it can misclassify non-option/invalid tokens as short options on reasonable inputs beyond the failing test, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement changed the logic from returning true if any character anywhere in optName is a valid short option to recognizing only a valid prefix of consecutive short options from the start of the token, stopping at the first invalid character or '='. This is necessary because the parser should classify a token as a short-option token only when it starts with valid short option(s), not merely because some later character happens to match an option.

**Verdict:** Because the OD refinement introduced necessary prefix-validation logic that the pre-patch lacked, the original patch was an overfitting fix. It passed the observed tests but remained semantically wrong on other plausible inputs.

**Before OD (overfitting):**
```diff
@@ -8,7 +8,13 @@
 
 // remove leading "-" and "=value"
-int pos = token.indexOf("=");
+int pos = token.indexOf('=');
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+
+// Check each character in optName to see if it's a valid short option
+for (char c : optName.toCharArray()) {
+if (options.hasShortOption(String.valueOf(c))) {
+return true;
 }
+}
+return false;
+}
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that isShortOption only checked the entire post-dash token as one short option name, so concatenated short-option tokens like "-ab" or mixed forms like "-t1" were not recognized by checking the first short option component.

**Why pre-patch overfits:** The pre-patch is too narrow: it treats every short-option token as if only the first character matters, ignoring the rest of the token entirely. That happens to satisfy the shown failing test, but it breaks reasonable cases such as a defined multi-character short option like "-t1" or "-SV" where the whole short option name should be recognized, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The pre-patch reduced every candidate token to just token.substring(1, 2), i.e. only the first character after '-', whereas the refinement added logic to scan successive characters until '=' or an invalid short option is reached. This is necessary because short-option parsing must recognize more than a single leading character in tokens such as concatenated short options and forms involving multi-character short options or attached values.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent rewrite, the pre-patch was an overfitting fix. It passed the tests by coincidence but would mis-handle valid inputs outside the specific failing case.

**Before OD (overfitting):**
```diff
@@ -8,6 +8,6 @@
 
 // remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
+int pos = token.indexOf('=');
+String optName = token.substring(1, 2);
 return options.hasShortOption(optName);
 // check for several concatenated short options
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that isShortOption only checked whether the entire post-dash token was a declared short option, so it failed to recognize tokens that begin with a valid short option but also contain concatenated options or an attached value.

**Why pre-patch overfits:** The pre-patch fixes the specific concatenated-short-options test by accepting any token whose characters after '-' are all individually known short options, but it drops the original support for whole short option names and for '=value' syntax. For example, a valid token like '-t1=foo' where 't1' is a short option would be rejected because the scan reaches '=' and stops, even though the method is supposed to recognize '-S=V' forms.

**OD refinement (NECESSARY):** The refinement added handling for '=' as a delimiter, so parsing stops before the attached value portion of a token like '-S=V' or '-SV1=V2'. This is necessary because the pre-patch scans every character in the raw token and therefore misclassifies valid short options with attached values when it reaches '=' or value characters.

**Verdict:** Because the refinement change is NECESSARY and adds missing delimiter logic required for correct short-option recognition with attached values, the pre-patch was not a complete semantic fix and was overfitting to the tested concatenation scenario.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,14 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
 // check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
 }
+}
+return i > 1;
+}
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that isShortOption only checks whether the whole post-dash token is a defined short option, so it fails to recognize tokens that begin with a valid short option but contain concatenated short options or attached data.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it never verifies that a longer token corresponds to any defined short option sequence. It would incorrectly return true for reasonable inputs like an unknown token such as '-xyz' or a long option-like token without '=', causing parsing behavior to diverge from the intended semantics outside the specific failing test.

**OD refinement (NECESSARY):** The pre-patch treated any dash-prefixed token longer than two characters and without '=' as a short option by returning true unconditionally. The refinement replaced that with actual validation of the token contents, checking character by character whether the token starts with one or more defined short options and stopping at '=' or the first invalid character; this is necessary to avoid misclassifying arbitrary long tokens as short options.

**Verdict:** Because the refinement introduced necessary missing validation logic rather than a cosmetic rewrite, the pre-patch was overfitting. Its unconditional acceptance of all multi-character dash tokens is semantically wrong and not an alternative correct implementation.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,12 @@
 }
 
+// Check for several concatenated short options
+if (token.indexOf('=') == -1 && token.length() > 2) {
+return true;
+}
+
 // remove leading "-" and "=value"
-int pos = token.indexOf("=");
+int pos = token.indexOf('=');
 String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
 return options.hasShortOption(optName);
-// check for several concatenated short options
 }
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that isShortOption only checks whether the whole post-dash token is a declared short option, so it fails to recognize tokens that begin with a valid short option but contain concatenated short options or attached data.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause; it overgeneralizes by treating every token of length greater than 2 and without '=' as a short option, regardless of whether any declared short option matches. That would misclassify reasonable inputs beyond the failing test, including unknown options or malformed tokens, so it passes the test by coincidence rather than by implementing the proper short-option recognition logic.

**OD refinement (NECESSARY):** The OD refinement replaced the pre-patch's unconditional acceptance of any dash-prefixed token longer than 2 without '=' with logic that actually validates characters as short options and stops at '='. This is necessary because the pre-patch returns true for many invalid tokens such as '-xyz' even when no corresponding short options exist, whereas the refined version checks that at least the leading short option structure is real.

**Verdict:** Because the refinement change is NECESSARY and adds missing validation logic absent from the pre-patch, the original patch is a true overfitting fix rather than an alternative correct implementation.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,12 @@
 }
 
+// Check for concatenated short options
+if (token.length() > 2 && !token.contains("=")) {
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
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that isShortOption only checks the whole post-dash token as one short option name, so concatenated short-option tokens like "-ab" are not recognized as short options starting with a valid short option.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause; it over-approximates by scanning all characters and accepting the token if any one of them is a known short option. On reasonable inputs beyond the test, such as a token whose first option character is invalid but a later one is valid, it would return true incorrectly and alter parsing behavior, so it is an overfitting fix.

**OD refinement (NECESSARY):** The OD refinement changed the pre-patch from returning true if any character anywhere after '-' is a known short option to recognizing only a valid prefix of concatenated short options and stopping at '='. This is necessary because the pre-patch misclassifies tokens such as "-xa" as short options merely because a later character is valid, whereas short-option parsing should depend on the beginning of the option sequence, not any arbitrary later character.

**Verdict:** Because the refinement introduced necessary semantic logic?봱equiring recognition of a valid leading short-option sequence rather than any matching later character?봳he pre-patch was not an alternative correct implementation. It passed the tests but remained semantically wrong on other plausible inputs.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,11 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
 // check for several concatenated short options
+for (int i = 1; i < token.length(); i++) {
+String optName = String.valueOf(token.charAt(i));
+if (options.hasShortOption(optName)) {
+return true;
 }
+}
+return false;
+}
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

**Root cause:** The bug is that `isShortOption` only checked the whole token body as one short option name and failed to recognize valid concatenated short options by considering the first short option before attached characters or values.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix. It ignores the original whole-name check and first-character-based concatenation logic, and would incorrectly return true for inputs like `-xa` when only `a` is a defined short option, even though the token does not start with a valid short option sequence. It also does not treat `=` as a delimiter, so its behavior is broader and semantically wrong beyond the triggering test.

**OD refinement (NECESSARY):** The refinement changed the logic from "return true if any character after '-' is a known short option" to "scan from the start of the token, stop at '=' or the first non-option character, and only accept the token if it begins with a valid short-option sequence." This is necessary because the pre-patch can misclassify tokens as short options merely because some later character happens to be a valid option, which is not the intended parser semantics.

**Verdict:** Because the refinement introduced necessary prefix-validation logic that the pre-patch lacked, the pre-patch was an overfitting fix rather than a genuinely correct one.

**Before OD (overfitting):**
```diff
@@ -7,8 +7,11 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+// Check for several concatenated short options
+for (int i = 1; i < token.length(); i++) {
+char optChar = token.charAt(i);
+if (options.hasShortOption(String.valueOf(optChar))) {
+return true;
 }
+}
+return false;
+}
```

**After OD (correct):**
```diff
@@ -7,8 +7,17 @@
 }
 
-// remove leading "-" and "=value"
-int pos = token.indexOf("=");
-String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
-return options.hasShortOption(optName);
-// check for several concatenated short options
+int i = 1;
+while (i < token.length()) {
+char c = token.charAt(i);
+if (c == '=') {
+// If we encounter '=', the rest of the string is the value, so we stop checking further
+break;
 }
+if (options.hasShortOption(String.valueOf(c))) {
+i++;
+} else {
+break;
+}
+}
+return i > 1;
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

## Closure-11

### Patch 1

**Root cause:** The bug is an incorrect early return for property accesses in assignment contexts, which skips the required null/undefined validation and property existence checking for getprop nodes.

**Why pre-patch overfits:** The pre-patch only changes behavior when parent.isAssign() is true, so it does not fully implement the ground-truth fix of eliminating the assignment-specific branch altogether. On non-assignment getprop expressions with null/undefined receivers, it still relies on a second expectNotNullOrUndefined call in a different branch rather than the correct single guard-and-return structure, making it semantically different and potentially incorrect beyond the observed tests.

**OD refinement (NECESSARY):** The refinement removes the special-case dependence on parent.isAssign() and makes the null/undefined guard apply to all property accesses before checkPropertyAccess. This is necessary because the buggy behavior was not limited to assignments on the parent node; the pre-patch still skips the intended control flow distinction for non-assignment property reads and can miss the proper early-return behavior after a failed null/undefined check.

**Verdict:** Because the OD refinement made a necessary semantic change?봱emoving the assignment-only condition and restoring the unconditional null/undefined guard?봳he pre-patch was not a complete fix of the root cause. Its logic remains assignment-specific and therefore overfits relative to the true fix.

**Before OD (overfitting):**
```diff
@@ -10,8 +10,7 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
+} else if (parent.isAssign() && !validator.expectNotNullOrUndefined(t, n, childType, "No properties on this expression", getNativeType(OBJECT_TYPE))) {
 return;
-} else if (validator.expectNotNullOrUndefined(t, n, childType,
-"No properties on this expression", getNativeType(OBJECT_TYPE))) {
+} else if (validator.expectNotNullOrUndefined(t, n, childType, "No properties on this expression", getNativeType(OBJECT_TYPE))) {
 checkPropertyAccess(childType, property.getString(), t, n);
 }
```

**After OD (correct):**
```diff
@@ -10,8 +10,7 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
+} else if (!validator.expectNotNullOrUndefined(t, n, childType, "No properties on this expression", getNativeType(OBJECT_TYPE))) {
 return;
-} else if (validator.expectNotNullOrUndefined(t, n, childType,
-"No properties on this expression", getNativeType(OBJECT_TYPE))) {
+} else {
 checkPropertyAccess(childType, property.getString(), t, n);
 }
```

**Ground truth fix:**
```diff
@@ -10,6 +10,4 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
-return;
 } else if (validator.expectNotNullOrUndefined(t, n, childType,
 "No properties on this expression", getNativeType(OBJECT_TYPE))) {
```

### Patch 2

**Root cause:** The bug is an incorrect early return for property accesses in assignment contexts, which suppresses the required null/undefined check and property existence validation on the left-hand side.

**Why pre-patch overfits:** The pre-patch is not equivalent to the ground-truth fix. For a non-assignment access like 'var y = x.prop;' where x is null or undefined, it would incorrectly bypass expectNotNullOrUndefined and go straight to checkPropertyAccess, changing diagnostics and potentially missing the intended warning; thus it would fail on reasonable inputs beyond the provided tests.

**OD refinement (NECESSARY):** The refinement removes the special-case dependence on parent.isAssign() and restores the original semantics: for every non-dict getprop, first require the base expression to be non-null/non-undefined, and only then check property access. The pre-patch changed behavior by skipping expectNotNullOrUndefined for non-assignment reads and by still calling checkPropertyAccess even when the base may be null/undefined in assignment cases, so the refinement adds back logic needed for general correctness.

**Verdict:** Because the refinement change is NECESSARY and restores missing validation for all getprop cases, the pre-patch was an overfitting fix rather than a semantically correct alternative.

**Before OD (overfitting):**
```diff
@@ -10,8 +10,6 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
-return;
-} else if (validator.expectNotNullOrUndefined(t, n, childType,
-"No properties on this expression", getNativeType(OBJECT_TYPE))) {
+} else if (parent.isAssign() && !validator.expectNotNullOrUndefined(t, n, childType, "No properties on this expression", getNativeType(OBJECT_TYPE))) {
+} else {
 checkPropertyAccess(childType, property.getString(), t, n);
 }
```

**After OD (correct):**
```diff
@@ -10,8 +10,5 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
-return;
-} else if (validator.expectNotNullOrUndefined(t, n, childType,
-"No properties on this expression", getNativeType(OBJECT_TYPE))) {
+} else if (validator.expectNotNullOrUndefined(t, n, childType, "No properties on this expression", getNativeType(OBJECT_TYPE))) {
 checkPropertyAccess(childType, property.getString(), t, n);
 }
```

**Ground truth fix:**
```diff
@@ -10,6 +10,4 @@
 if (childType.isDict()) {
 report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-} else if (n.getJSType() != null && parent.isAssign()) {
-return;
 } else if (validator.expectNotNullOrUndefined(t, n, childType,
 "No properties on this expression", getNativeType(OBJECT_TYPE))) {
```

---

## Closure-114

### Patch 1

**Root cause:** The bug is that assignments used as the callee of a CALL expression are incorrectly recorded in the dependency scope of the assigned name, causing the assigned variable to be treated as needed when only the RHS function value is actually used.

**Why pre-patch overfits:** The pre-patch does not fix the root cause; it simply avoids processing assignments whose left-hand side name is literally "fun". It would fail on any semantically identical input using a different variable name, such as `(g = function(){ x; })();`, where the same incorrect dependency recording would still occur.

**OD refinement (NECESSARY):** The refinement removes the hardcoded exclusion for the specific variable name "fun" and replaces it with structural logic that skips recording dependency scope whenever the assignment is part of a call expression callee. This is necessary because the bug depends on AST context, not on any particular identifier, so the pre-patch only masks the tested case instead of fixing the underlying condition.

**Verdict:** Because the OD refinement introduced necessary context-sensitive logic absent from the pre-patch, the original patch was a name-specific overfit rather than a genuine fix. The refined patch addresses the actual call-callee assignment case, while the pre-patch only special-cases one test identifier.

**Before OD (overfitting):**
```diff
@@ -3,5 +3,5 @@
 Node parent = n.getParent();
 NameInformation ns = createNameInformation(t, nameNode);
-if (ns != null) {
+if (ns != null && !nameNode.getString().equals("fun")) {
 if (parent.isFor() && !NodeUtil.isForIn(parent)) {
 // Patch for assignments that appear in the init,
```

**After OD (correct):**
```diff
@@ -26,6 +26,19 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!isPartOfCallExpression(n)) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
+
+private boolean isPartOfCallExpression(Node n) {
+Node current = n;
+while (current != null) {
+if (current.getParent() != null && current.getParent().isCall() && current.getParent().getFirstChild() == current) {
+return true;
+}
+current = current.getParent();
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,5 @@
 recordDepScope(nameNode, ns);
 }
-} else {
+} else if (!(parent.isCall() && parent.getFirstChild() == n)) {
 // The rhs of the assignment is the caller, so it's used by the
 // context. Don't associate it w/ the lhs.
```

### Patch 2

**Root cause:** The bug is that recordAssignment incorrectly records dependency scope when an assignment expression is used as the callee of a CALL, causing the assigned name to be treated as needed and preventing its removal.

**Why pre-patch overfits:** The pre-patch does not fix the root cause correctly: it disables all handling for assignments whose left-hand side is a simple name, which is far broader than the intended fix and unrelated to call-callee context. That means many normal assignments to names outside calls would no longer record dependency scope, so the patch would produce incorrect behavior on reasonable unseen inputs even though it passes the given tests.

**OD refinement (NECESSARY):** The refinement removes the pre-patch's blanket guard `!nameNode.isName()` and instead adds logic to skip recording dependency scope specifically when the assignment node is part of a call expression used as the callee. This is necessary because the bug depends on the assignment's syntactic context, not on whether the left-hand side is a NAME node; the pre-patch suppresses processing for ordinary name assignments everywhere, which is semantically wrong.

**Verdict:** Because the OD refinement introduced necessary context-sensitive logic that the pre-patch lacked, and the pre-patch's `!nameNode.isName()` condition is an overbroad, semantically incorrect substitute, the original patch is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -3,15 +3,6 @@
 Node parent = n.getParent();
 NameInformation ns = createNameInformation(t, nameNode);
-if (ns != null) {
+if (ns != null && !nameNode.isName()) {
 if (parent.isFor() && !NodeUtil.isForIn(parent)) {
-// Patch for assignments that appear in the init,
-// condition or iteration part of a FOR loop.  Without
-// this change, all 3 of those parts try to claim the for
-// loop as their dependency scope.  The last assignment in
-// those three fields wins, which can result in incorrect
-// reference edges between referenced and assigned variables.
-//
-// TODO(user) revisit the dependency scope calculation
-// logic.
 if (parent.getFirstChild().getNext() != n) {
 recordDepScope(recordNode, ns);
@@ -20,10 +11,4 @@
 }
 } else {
-// The rhs of the assignment is the caller, so it's used by the
-// context. Don't associate it w/ the lhs.
-// FYI: this fixes only the specific case where the assignment is the
-// caller expression, but it could be nested deeper in the caller and
-// we would still get a bug.
-// See testAssignWithCall2 for an example of this.
 recordDepScope(recordNode, ns);
 }
```

**After OD (correct):**
```diff
@@ -26,6 +26,19 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!isPartOfCallExpression(n)) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
+
+private boolean isPartOfCallExpression(Node n) {
+Node current = n;
+while (current != null) {
+if (current.getParent() != null && current.getParent().isCall() && current.getParent().getFirstChild() == current) {
+return true;
+}
+current = current.getParent();
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,5 @@
 recordDepScope(nameNode, ns);
 }
-} else {
+} else if (!(parent.isCall() && parent.getFirstChild() == n)) {
 // The rhs of the assignment is the caller, so it's used by the
 // context. Don't associate it w/ the lhs.
```

### Patch 3

**Root cause:** The bug is that assignments used as the callee of a CALL expression are incorrectly recorded in the dependency scope, causing the assigned name to be treated as needed when only the RHS function value is actually used.

**Why pre-patch overfits:** The pre-patch does not fix the root cause; it merely special-cases the identifier `fun`, which happens to satisfy the failing test. It would still behave incorrectly for reasonable variants such as `(g = function(){ x; })();` or other assignments in callee position, so it is an overfitting patch rather than a general semantic fix.

**OD refinement (NECESSARY):** The refinement replaces a hardcoded variable-name check (`!"fun".equals(nameNode.getString())`) with a structural AST check that skips dependency recording whenever the assignment node is part of a call expression callee position. This is necessary because the bug is about syntactic context, not about a specific identifier name, so the pre-patch only handles the tested example and misses equivalent cases with other variable names or nested call-callee assignments.

**Verdict:** Because the refinement change is NECESSARY and adds missing context-sensitive logic that the pre-patch does not otherwise implement, the original patch is a true overfitting fix and therefore incorrect.

**Before OD (overfitting):**
```diff
@@ -26,6 +26,8 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!"fun".equals(nameNode.getString())) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -26,6 +26,19 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!isPartOfCallExpression(n)) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
+
+private boolean isPartOfCallExpression(Node n) {
+Node current = n;
+while (current != null) {
+if (current.getParent() != null && current.getParent().isCall() && current.getParent().getFirstChild() == current) {
+return true;
+}
+current = current.getParent();
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,5 @@
 recordDepScope(nameNode, ns);
 }
-} else {
+} else if (!(parent.isCall() && parent.getFirstChild() == n)) {
 // The rhs of the assignment is the caller, so it's used by the
 // context. Don't associate it w/ the lhs.
```

### Patch 4

**Root cause:** The bug is that assignments used as the callee of a call expression were incorrectly recorded as dependencies of the assigned name, causing the analyzer to think the assigned variable was needed and preserve it when it should be removable.

**Why pre-patch overfits:** The pre-patch does not fix the root cause correctly: for almost all non-FOR assignments with non-null name information, it records dependency scope on the lhs nameNode instead of the recordNode, which is the opposite of the ground-truth logic except for the special call-callee case. It would therefore mis-handle many ordinary assignments and FOR-loop cases on inputs beyond the observed test, so its test success is accidental/overfitting.

**OD refinement (NECESSARY):** The refinement restored the original FOR-loop special handling and, more importantly, changed the non-FOR case from always recording dependency on the lhs/nameNode to conditionally skipping dependency recording when the assignment is part of a call expression. This is necessary because the pre-patch reverses the intended behavior for ordinary assignments and also breaks the FOR-loop condition/update distinction, so it does not implement the actual dependency-scope rule.

**Verdict:** Because the refinement change is NECESSARY and adds missing semantic logic that the pre-patch does not achieve by any equivalent means, the original patch is a true overfitting patch rather than a correct alternative fix.

**Before OD (overfitting):**
```diff
@@ -5,25 +5,10 @@
 if (ns != null) {
 if (parent.isFor() && !NodeUtil.isForIn(parent)) {
-// Patch for assignments that appear in the init,
-// condition or iteration part of a FOR loop.  Without
-// this change, all 3 of those parts try to claim the for
-// loop as their dependency scope.  The last assignment in
-// those three fields wins, which can result in incorrect
-// reference edges between referenced and assigned variables.
-//
-// TODO(user) revisit the dependency scope calculation
-// logic.
-if (parent.getFirstChild().getNext() != n) {
+if (parent.getFirstChild().getNext() != n && !nameNode.isVar()) {
 recordDepScope(recordNode, ns);
 } else {
 recordDepScope(nameNode, ns);
 }
-} else {
-// The rhs of the assignment is the caller, so it's used by the
-// context. Don't associate it w/ the lhs.
-// FYI: this fixes only the specific case where the assignment is the
-// caller expression, but it could be nested deeper in the caller and
-// we would still get a bug.
-// See testAssignWithCall2 for an example of this.
+} else if (!nameNode.isVar()) {
 recordDepScope(recordNode, ns);
 }
```

**After OD (correct):**
```diff
@@ -26,6 +26,19 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!isPartOfCallExpression(n)) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
+
+private boolean isPartOfCallExpression(Node n) {
+Node current = n;
+while (current != null) {
+if (current.getParent() != null && current.getParent().isCall() && current.getParent().getFirstChild() == current) {
+return true;
+}
+current = current.getParent();
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,5 @@
 recordDepScope(nameNode, ns);
 }
-} else {
+} else if (!(parent.isCall() && parent.getFirstChild() == n)) {
 // The rhs of the assignment is the caller, so it's used by the
 // context. Don't associate it w/ the lhs.
```

### Patch 5

**Root cause:** The bug is that assignments used as the callee of a CALL expression were incorrectly recorded as dependencies of the assigned name, causing the assigned variable to appear used when only the RHS function value is actually invoked.

**Why pre-patch overfits:** The pre-patch does not fix the root cause; it merely suppresses dependency recording when the assigned variable happens to be named `fun`. Any equivalent program using another name, such as `(g = function(){ x; })();`, would still be handled incorrectly, so the patch is clearly overfitted to the failing test input.

**OD refinement (NECESSARY):** The refinement replaced the hardcoded special case `!nameNode.getString().equals("fun")` with structural logic that detects whether the assignment node is part of a call expression and skips dependency recording in that case. This is necessary because the bug depends on AST context, not on a specific variable name, so the pre-patch only fixes the tested example and not the general case.

**Verdict:** Because the refinement introduced necessary missing logic that generalizes from a test-specific name check to the actual syntactic condition, the pre-patch was not semantically correct and was a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -5,13 +5,4 @@
 if (ns != null) {
 if (parent.isFor() && !NodeUtil.isForIn(parent)) {
-// Patch for assignments that appear in the init,
-// condition or iteration part of a FOR loop.  Without
-// this change, all 3 of those parts try to claim the for
-// loop as their dependency scope.  The last assignment in
-// those three fields wins, which can result in incorrect
-// reference edges between referenced and assigned variables.
-//
-// TODO(user) revisit the dependency scope calculation
-// logic.
 if (parent.getFirstChild().getNext() != n) {
 recordDepScope(recordNode, ns);
@@ -20,12 +11,8 @@
 }
 } else {
-// The rhs of the assignment is the caller, so it's used by the
-// context. Don't associate it w/ the lhs.
-// FYI: this fixes only the specific case where the assignment is the
-// caller expression, but it could be nested deeper in the caller and
-// we would still get a bug.
-// See testAssignWithCall2 for an example of this.
+if (!nameNode.getString().equals("fun")) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -26,6 +26,19 @@
 // we would still get a bug.
 // See testAssignWithCall2 for an example of this.
+if (!isPartOfCallExpression(n)) {
 recordDepScope(recordNode, ns);
 }
 }
 }
+}
+
+private boolean isPartOfCallExpression(Node n) {
+Node current = n;
+while (current != null) {
+if (current.getParent() != null && current.getParent().isCall() && current.getParent().getFirstChild() == current) {
+return true;
+}
+current = current.getParent();
+}
+return false;
+}
```

**Ground truth fix:**
```diff
@@ -19,5 +19,5 @@
 recordDepScope(nameNode, ns);
 }
-} else {
+} else if (!(parent.isCall() && parent.getFirstChild() == n)) {
 // The rhs of the assignment is the caller, so it's used by the
 // context. Don't associate it w/ the lhs.
```

---

## Closure-128

**Root cause:** The bug is that isSimpleNumber incorrectly rejects the valid numeric property name "0" by requiring the first character not be '0', instead of allowing the single-digit zero while still rejecting multi-digit numbers with leading zeros.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test because it makes "0" count as a simple number, but it overgeneralizes and treats every non-empty all-digit string as simple. That would mis-handle reasonable inputs beyond the test, such as object keys like "00" or "012", which should remain quoted rather than being emitted as simple numeric property names.

**OD refinement (NECESSARY):** The OD refinement restored the missing leading-zero rule by changing the return condition from accepting any non-empty digit string to accepting only either a single digit or a digit string whose first character is not '0'. This is necessary because the pre-patch wrongly classifies strings like "00" and "012" as simple numbers, which changes code generation semantics for numeric-looking property names with leading zeros; the added null/empty guard is not the key issue, but the leading-zero logic is.

**Verdict:** Because the refinement change is NECESSARY and restores logic that the pre-patch removed entirely, the pre-patch does not correctly implement the root cause fix. It passes the observed test by coincidence while remaining semantically wrong for other numeric strings with leading zeros.

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
+if (s == null || s.isEmpty()) {
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

## Closure-161

### Patch 1

**Root cause:** The method folds array element access even when the GETELEM node is being used as an assignment target, which is invalid because the array literal is only a temporary l-value context and should not be constant-folded.

**Why pre-patch overfits:** The pre-patch does not fix the root cause; it merely avoids one manifestation by returning early when the array literal has no children. It would still incorrectly fold non-empty array literals used as assignment targets, such as `[0][0] = 1`, and therefore remains semantically wrong on reasonable inputs beyond the observed failing test.

**OD refinement (NECESSARY):** The refinement added an early return when the array access node is part of an assignment operation, preventing folding of expressions like `[][1] = 1` and similar assignment-target cases. This is the actual semantic guard present in the ground-truth fix; the pre-patch instead only special-cased empty array literals, which does not address the broader assignment-target bug.

**Verdict:** Because the OD refinement introduced necessary assignment-target detection that was absent from the pre-patch, the original patch was overfitting to the empty-array test case rather than implementing the real fix.

**Before OD (overfitting):**
```diff
@@ -23,4 +23,8 @@
 }
 
+if (left.getFirstChild() == null) {
+return n;
+}
+
 Node elem = left.getFirstChild();
 for (int i = 0; elem != null && i < intIndex; i++) {
```

**After OD (correct):**
```diff
@@ -4,4 +4,8 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+
+if (parent != null && NodeUtil.isAssignmentOp(parent)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

**Ground truth fix:**
```diff
@@ -4,4 +4,7 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+if (isAssignmentTarget(n)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

### Patch 2

**Root cause:** The bug is that array access folding runs even when the GETELEM node is being used as an assignment target, where folding is invalid and can trigger incorrect out-of-bounds errors or wrong transformations.

**Why pre-patch overfits:** The pre-patch fixes the specific failing case `[][1] = 1` by handling plain assignment, but it does not fully fix the root cause because it still folds when the array access is the target of compound assignment operations such as `[][0] += 1`. On such reasonable inputs beyond the provided test, it would still behave incorrectly, so its success is too narrow.

**OD refinement (NECESSARY):** The refinement broadens the guard from only `parent.getType() == Token.ASSIGN` to `parent != null && NodeUtil.isAssignmentOp(parent)`, which covers all assignment operators, not just simple `=`. This is necessary because the comment and ground-truth fix require skipping folding for any assignment target usage, including compound assignments like `+=`.

**Verdict:** Because the refinement change is NECESSARY and adds missing assignment-target coverage that the pre-patch lacks, the pre-patch is an overfitting fix rather than a complete semantic repair.

**Before OD (overfitting):**
```diff
@@ -4,4 +4,8 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+
+if (parent.getType() == Token.ASSIGN) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

**After OD (correct):**
```diff
@@ -4,4 +4,8 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+
+if (parent != null && NodeUtil.isAssignmentOp(parent)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

**Ground truth fix:**
```diff
@@ -4,4 +4,7 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+if (isAssignmentTarget(n)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

### Patch 3

**Root cause:** The method folds array element access even when the GETELEM node is being used as an assignment target, where folding is invalid and can trigger bogus out-of-bounds errors.

**Why pre-patch overfits:** The pre-patch does not fix the root cause correctly; instead, it effectively disables array-access folding altogether by always returning `n` after index validation. That avoids the failing test by coincidence, but it breaks valid constant folding for ordinary reads such as array literal accesses that should be reduced, so it would be incorrect on reasonable inputs beyond the observed test.

**OD refinement (NECESSARY):** The refinement restores the real folding logic and adds an early guard for assignment operations so expressions like `[][1] = 1` are left unchanged only in the assignment-target case. It also preserves the original behavior for non-assignment reads, including reporting out-of-bounds errors and replacing the access with the folded element when valid; this logic is missing from the pre-patch.

**Verdict:** Because the refinement change is NECESSARY and adds missing assignment-target handling while restoring required folding semantics, the pre-patch is a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -28,18 +28,10 @@
 }
 
+// If elem is null, it means we are trying to assign to an index that doesn't exist yet.
+// This is a valid operation and should not be folded.
 if (elem == null) {
-error(INDEX_OUT_OF_BOUNDS_ERROR, right);
 return n;
 }
 
-if (elem.getType() == Token.EMPTY) {
-elem = NodeUtil.newUndefinedNode(elem);
-} else {
-left.removeChild(elem);
+return n;
 }
-
-// Replace the entire GETELEM with the value
-n.getParent().replaceChild(n, elem);
-reportCodeChange();
-return elem;
-}
```

**After OD (correct):**
```diff
@@ -4,4 +4,8 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+
+if (parent != null && NodeUtil.isAssignmentOp(parent)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

**Ground truth fix:**
```diff
@@ -4,4 +4,7 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+if (isAssignmentTarget(n)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

### Patch 4

**Root cause:** The bug is that array access folding is performed even when the GETELEM node is being used as an assignment target, where folding is invalid and can trigger bogus out-of-bounds errors or incorrect rewrites.

**Why pre-patch overfits:** The pre-patch only handles the specific case where the parent is `Token.ASSIGN`, so it fixes `[][1] = 1` but still mishandles reasonable related inputs like `[][0] += 1`, which the method comment explicitly identifies as part of the bug. It therefore does not fully fix the root cause and remains vulnerable to incorrect behavior outside the observed test.

**OD refinement (NECESSARY):** The refinement changed the guard from only skipping plain `ASSIGN` parents to skipping all assignment operations via `NodeUtil.isAssignmentOp(parent)`, and it moved that check before any index validation/error reporting. This is necessary because assignment targets include compound assignments such as `+=`, and those should also not be folded or diagnosed as invalid array accesses.

**Verdict:** Because the refinement added necessary missing logic for assignment targets beyond simple `=` and the pre-patch did not achieve that behavior by any alternative means, the pre-patch was an overfitting fix rather than a semantically complete one.

**Before OD (overfitting):**
```diff
@@ -23,4 +23,9 @@
 }
 
+// If the array is being assigned to, do not attempt to fold it
+if (parent.getType() == Token.ASSIGN) {
+return n;
+}
+
 Node elem = left.getFirstChild();
 for (int i = 0; elem != null && i < intIndex; i++) {
```

**After OD (correct):**
```diff
@@ -4,4 +4,8 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+
+if (parent != null && NodeUtil.isAssignmentOp(parent)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

**Ground truth fix:**
```diff
@@ -4,4 +4,7 @@
 // acting as a temporary we can't fold it here:
 //    "[][0] += 1"
+if (isAssignmentTarget(n)) {
+return n;
+}
 
 if (right.getType() != Token.NUMBER) {
```

---

## Closure-7

### Patch 1

**Root cause:** When refining an ObjectType for typeof == "function", the buggy code returned null for all non-matching object cases, incorrectly removing ordinary objects on the false branch instead of preserving them unless they are known function subtypes.

**Why pre-patch overfits:** The pre-patch fixes the observed failing tests by preserving ordinary objects, but it does so too broadly: for value.equals("function") and resultEqualsValue == false, it always returns type, even when type itself is a function subtype. That would be incorrect on reasonable inputs such as a known function object under a false typeof-function condition, so it does not correctly implement the root cause.

**OD refinement (NECESSARY):** The OD refinement changed the behavior for the typeof "function" case when resultEqualsValue is false: instead of unconditionally returning type, it falls through to matchesExpectation("object"). This is necessary because the pre-patch wrongly treats every object as surviving typeof != "function", including function subtypes that should be filtered out, whereas the refined version restores the intended branch-sensitive behavior.

**Verdict:** Because the refinement introduced a necessary semantic change on the false branch, the pre-patch was not equivalent to the correct behavior. It passed the given tests by over-approximating and would mis-handle function-typed objects outside those tests.

**Before OD (overfitting):**
```diff
@@ -2,5 +2,5 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
+return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : type;
 // Objects are restricted to "Function", subtypes are left
 // Only filter out subtypes of "function"
```

**After OD (correct):**
```diff
@@ -2,7 +2,7 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
-// Objects are restricted to "Function", subtypes are left
-// Only filter out subtypes of "function"
+if (resultEqualsValue) {
+return ctorType.isSubtype(type) ? ctorType : null;
+}
 }
 return matchesExpectation("object") ? type : null;
```

**Ground truth fix:**
```diff
@@ -2,7 +2,11 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
+if (resultEqualsValue) {
 // Objects are restricted to "Function", subtypes are left
+return ctorType.getGreatestSubtype(type);
+} else {
 // Only filter out subtypes of "function"
+return type.isSubtype(ctorType) ? null : type;
+}
 }
 return matchesExpectation("object") ? type : null;
```

### Patch 2

**Root cause:** When refining an object type for `typeof x == 'function'`, the method mishandles the false branch by returning `null` instead of preserving non-function object types, and more generally should only exclude actual function subtypes.

**Why pre-patch overfits:** The pre-patch fixes the observed failing tests by preserving the original object type on the false branch for `typeof == 'function'`, but it does so too broadly. For inputs where `value` is `function`, `resultEqualsValue` is false, and the surrounding expectation is not `object` (or should otherwise reject the object type), it would incorrectly return `type` instead of delegating to the normal expectation logic, so it does not correctly implement the method's intended semantics.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch behavior that unconditionally returns `type` whenever `value.equals("function")` and `resultEqualsValue` is false. This is semantically important because the false branch should still respect `matchesExpectation("object")`; otherwise `typeof x != 'function'` incorrectly keeps object types even in contexts where the expected typeof is not `object`.

**Verdict:** Because the refinement change is NECESSARY and restores missing semantic filtering on the false branch, the pre-patch was not a fully correct fix. It passed tests by over-approximating the false case rather than correctly handling all reasonable inputs.

**Before OD (overfitting):**
```diff
@@ -2,5 +2,5 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
+return resultEqualsValue ? (ctorType.isSubtype(type) ? ctorType : null) : type;
 // Objects are restricted to "Function", subtypes are left
 // Only filter out subtypes of "function"
```

**After OD (correct):**
```diff
@@ -2,7 +2,7 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
-// Objects are restricted to "Function", subtypes are left
-// Only filter out subtypes of "function"
+if (resultEqualsValue) {
+return ctorType.isSubtype(type) ? ctorType : null;
+}
 }
 return matchesExpectation("object") ? type : null;
```

**Ground truth fix:**
```diff
@@ -2,7 +2,11 @@
 if (value.equals("function")) {
 JSType ctorType = getNativeType(U2U_CONSTRUCTOR_TYPE);
-return resultEqualsValue && ctorType.isSubtype(type) ? ctorType : null;
+if (resultEqualsValue) {
 // Objects are restricted to "Function", subtypes are left
+return ctorType.getGreatestSubtype(type);
+} else {
 // Only filter out subtypes of "function"
+return type.isSubtype(ctorType) ? null : type;
+}
 }
 return matchesExpectation("object") ? type : null;
```

---

## Compress-28

### Patch 1

**Root cause:** The method treated an unexpected end-of-stream while bytes were still expected for the current TAR entry as normal EOF instead of signaling a truncated archive with an IOException.

**Why pre-patch overfits:** The pre-patch does address the main failing test by throwing on `-1`, but it also adds semantically wrong behavior: it treats any zero-byte read before `entrySize` is reached as truncation. On reasonable non-truncated inputs using an underlying stream that can return 0 transiently, it would spuriously throw IOException, so it does not correctly implement the root cause in a general way.

**OD refinement (NECESSARY):** The refinement removed the pre-patch's extra truncation condition on `totalRead == 0` and only throws when `totalRead == -1` before the entry is fully consumed, while also avoiding `entryOffset` updates for zero-byte reads. This is necessary because a zero-byte read is not equivalent to EOF for `InputStream.read(byte[], int, int)` and the pre-patch would incorrectly throw on streams that legally return 0 without being truncated.

**Verdict:** Because the OD refinement made a necessary semantic correction?봡istinguishing true EOF (`-1`) from a zero-byte read?봳he pre-patch was not already correct. Its added `totalRead == 0` truncation check is overfitting/wrong behavior beyond the tested case.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || (entryOffset < entrySize && totalRead == 0)) {
 hasHitEOF = true;
+throw new IOException("Truncated entry detected");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -15,7 +15,8 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated entry detected");
+} else if (totalRead > 0) {
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

### Patch 2

**Root cause:** The method treated an unexpected end of the underlying stream during an entry as normal EOF instead of signaling a truncated TAR entry with an IOException.

**Why pre-patch overfits:** The pre-patch does not correctly implement the root cause fix because it throws IOException whenever is.read returns -1, even in the legitimate case where no more bytes are requested/available at the end of an entry. It would therefore misbehave on reasonable inputs beyond the triggering test, and its added totalRead == 0 truncation check is also semantically unsupported.

**OD refinement (NECESSARY):** The refinement changed the exception condition from always throwing on any -1/0 read to throwing only when -1 occurs before the declared entry size has been fully consumed. This is necessary because the pre-patch incorrectly turns a normal end-of-entry read with totalRead == -1 into an exception, and also treats a 0-byte read as truncation, which is not the intended TAR truncation condition.

**Verdict:** Because the OD refinement made a necessary semantic correction to when truncation should be reported, the pre-patch was not already correct. It overfits by satisfying the failing test while introducing incorrect exception behavior for valid end-of-entry cases.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || (entryOffset < entrySize && totalRead == 0)) {
 hasHitEOF = true;
+throw new IOException("Truncated entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -16,4 +16,7 @@
 
 if (totalRead == -1) {
+if (entryOffset < entrySize) {
+throw new IOException("Truncated entry");
+}
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

### Patch 3

**Root cause:** The method treated an unexpected end of the underlying stream during an entry as normal EOF instead of throwing an IOException for a truncated TAR entry.

**Why pre-patch overfits:** The pre-patch does address the failing test's scenario of truncated input, but it is semantically too broad: it throws whenever is.read returns -1, even if the stream is at a valid boundary rather than truncated. It can therefore mis-handle reasonable inputs outside the test suite, especially around zero-length reads or end-of-archive conditions, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement changed the logic so that an IOException is thrown only when totalRead == -1 while entryOffset is still less than entrySize, and normal EOF is handled by merely setting hasHitEOF. This is necessary because the pre-patch incorrectly throws on any -1, including the legitimate case where no bytes remain to be read for the entry (for example when numToRead becomes 0), and it also adds an unsupported totalRead == 0 truncation condition.

**Verdict:** Because the OD refinement introduced necessary missing logic to distinguish true truncation from normal EOF, the pre-patch was not already correct. Its broader exception condition makes it an overfitting fix rather than a semantically sound one.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || entryOffset < entrySize) {
 hasHitEOF = true;
+throw new IOException("Truncated tar entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -15,8 +15,10 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
+throw new IOException("Truncated tar entry");
+} else if (totalRead != -1) {
+entryOffset += totalRead;
+} else {
 hasHitEOF = true;
-} else {
-entryOffset += totalRead;
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

### Patch 4

**Root cause:** The method treated an unexpected end-of-stream inside a tar entry as normal EOF instead of throwing an IOException for a truncated archive.

**Why pre-patch overfits:** The pre-patch does address the tested truncated-entry case, but it over-approximates truncation by throwing whenever a read returns fewer bytes than requested before entryOffset reaches entrySize. On reasonable valid inputs or stream implementations that legally return short reads, it would falsely report truncation, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement removed the extra condition that treated any short read (totalRead < numToRead while still inside the entry) as truncation, and limited truncation detection to the real error case: is.read returning -1 before the entry's declared size has been consumed. This is necessary because InputStream.read is allowed to return fewer bytes than requested without indicating EOF, so the pre-patch would incorrectly throw on valid partial reads.

**Verdict:** Because the OD refinement made a NECESSARY semantic correction to avoid misclassifying valid short reads as truncation, the pre-patch was an overfitting/incorrect fix rather than an alternative correct implementation.

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
@@ -15,7 +15,8 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated entry detected");
+} else if (totalRead > 0) {
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

### Patch 5

**Root cause:** The bug is that read() treats an unexpected underlying EOF during a non-empty remaining TAR entry as normal end-of-stream instead of throwing an IOException for a truncated archive.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause: it throws on any partial positive read, which is legal for many InputStreams, and it still fails to throw when the stream returns -1 before the entry is complete. Thus it can both reject valid inputs and miss genuine truncation on reasonable inputs beyond the observed test behavior.

**OD refinement (NECESSARY):** The refinement changes truncation detection from "short read but not -1" to "underlying read returned -1 before the entry's declared size was fully consumed." This is necessary because InputStream.read(byte[], ...) is allowed to return fewer bytes than requested without indicating truncation, while a -1 before entryOffset reaches entrySize is the real error condition.

**Verdict:** Because the refinement introduced necessary missing logic for detecting true truncation and the pre-patch's condition is semantically wrong rather than an alternative correct implementation, the original patch was an overfitting/incorrect fix.

**Before OD (overfitting):**
```diff
@@ -15,5 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead < numToRead && totalRead != -1) {
+throw new IOException("Truncated entry");
+} else if (totalRead == -1) {
 hasHitEOF = true;
 } else {
```

**After OD (correct):**
```diff
@@ -15,7 +15,8 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated entry detected");
+} else if (totalRead > 0) {
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

### Patch 6

**Root cause:** The method treats an unexpected end-of-stream while there are still bytes remaining in the current TAR entry as normal EOF instead of signaling a truncated archive with an IOException.

**Why pre-patch overfits:** The pre-patch is semantically wrong because it throws whenever a read returns fewer bytes than requested and the entry is not yet complete, even though short reads are legal and can occur on many streams without truncation. It also computes 'entryOffset + bytesRead' with bytesRead == -1, making its truncation test tied to an invalid arithmetic shortcut rather than the actual EOF condition, so it can misclassify valid inputs beyond the given test.

**OD refinement (NECESSARY):** The refinement moves truncation detection from the condition 'bytesRead < numToRead' to the correct condition 'is.read(...) returned -1 before entryOffset reached entrySize', and only advances entryOffset for positive reads. This is necessary because InputStream.read is allowed to return fewer bytes than requested without indicating truncation, while -1 is the real signal of premature EOF.

**Verdict:** Because the refinement introduced necessary missing logic for correct EOF-based truncation detection, the pre-patch was an overfitting/incorrect fix rather than an alternative correct implementation.

**Before OD (overfitting):**
```diff
@@ -12,5 +12,9 @@
 numToRead = Math.min(numToRead, available());
 
-totalRead = is.read(buf, offset, numToRead);
+int bytesRead = is.read(buf, offset, numToRead);
+if (bytesRead < numToRead && entryOffset + bytesRead < entrySize) {
+throw new IOException("Truncated entry detected");
+}
+totalRead = bytesRead;
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
+throw new IOException("Truncated entry detected");
+} else if (totalRead > 0) {
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

### Patch 7

**Root cause:** The method treated an unexpected end-of-stream while bytes were still expected for the current TAR entry as normal EOF instead of signaling a truncated archive with an IOException.

**Why pre-patch overfits:** The pre-patch does catch the tested truncated-entry case, but it is semantically too aggressive because InputStream.read is allowed to return fewer bytes than requested without indicating EOF. On reasonable inputs such as slow, buffered, or chunked underlying streams that legally produce short positive reads, the pre-patch would incorrectly throw IOException for a non-truncated TAR entry, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement removed the extra condition that treated any short read (totalRead < numToRead before entry end) as truncation and restricted truncation detection to the real error case: is.read returning -1 before the entry is fully consumed. It also avoided updating entryOffset on non-positive reads, which matches the intended stream semantics and the ground-truth fix.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch was not merely differently structured; it had incorrect truncation logic that would misclassify valid short reads as archive corruption. Therefore the original patch was an overfitting/semantically wrong fix.

**Before OD (overfitting):**
```diff
@@ -15,6 +15,7 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 || (totalRead < numToRead && entryOffset + totalRead < entrySize)) {
 hasHitEOF = true;
+throw new IOException("Truncated TAR entry");
 } else {
 entryOffset += totalRead;
```

**After OD (correct):**
```diff
@@ -15,7 +15,8 @@
 count(totalRead);
 
-if (totalRead == -1) {
+if (totalRead == -1 && entryOffset < entrySize) {
 hasHitEOF = true;
-} else {
+throw new IOException("Truncated entry detected");
+} else if (totalRead > 0) {
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

**Root cause:** The bug was that parseOctal incorrectly treated an embedded NUL inside the significant digit region as a terminator instead of rejecting it as an invalid octal field.

**Why pre-patch overfits:** The pre-patch does address the embedded-NUL symptom, but it does so with extra logic not present in the true fix: it throws immediately for any interior space or NUL rather than relying on the standard invalid-digit validation after trimming. While this likely behaves the same on many inputs, it is not the same root-cause fix as the GT patch and introduces stricter special-case behavior beyond what was needed.

**OD refinement (NECESSARY):** The OD refinement removed the newly added special-case rejection of embedded spaces and NULs before the normal digit-range check. This was necessary because, after leading spaces are skipped and trailing spaces/NULs are trimmed, any remaining embedded space or NUL is already invalid and correctly rejected by the existing range check; the pre-patch changed semantics by explicitly rejecting spaces/NULs in a way that is not the same as the ground-truth fix.

**Verdict:** Because the refinement change is NECESSARY and removes semantically different special-case logic, the pre-patch was not already the genuine fix. The confirmed-correct refined patch matches the actual root-cause repair, whereas the pre-patch overfits by adding unnecessary rejection logic.

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

**Root cause:** The bug is that parseOctal incorrectly stops parsing when it encounters an embedded NUL inside the significant field instead of treating any non-octal character after trimming leading spaces and trailing space/NUL padding as invalid.

**Why pre-patch overfits:** The pre-patch does not implement the correct semantics: it throws too early for any trailing padding because it scans for spaces/NULs before trimming them, so valid inputs like octal digits followed by the required trailing NUL or space would be rejected. It also adds an unnecessary empty-field exception and retains the silent break on NUL, so it would mis-handle reasonable inputs beyond the tests and is not an alternative correct fix.

**OD refinement (NECESSARY):** The refinement removes the pre-patch's added early scan that rejects any space or NUL anywhere before trailing trimming, removes the extra end<=start exception, and most importantly removes the in-loop special case that breaks on currentByte == 0. This is necessary because valid tar fields may contain trailing space/NUL padding that must be trimmed first, and after trimming, any remaining embedded NUL/space should be rejected by the normal non-octal check rather than by ad hoc prechecks or silent termination.

**Verdict:** The OD refinement is NECESSARY because it restores the proper parse order and validation semantics required by the ground-truth fix. Therefore the pre-refinement patch was a true overfitting/semantically wrong patch rather than an already-correct fix.

**Before OD (overfitting):**
```diff
@@ -21,4 +21,12 @@
 }
 
+// Check for embedded spaces or NULs
+for (int i = start; i < end; i++) {
+if (buffer[i] == ' ' || buffer[i] == 0) {
+throw new IllegalArgumentException(
+exceptionMessage(buffer, offset, length, i, buffer[i]));
+}
+}
+
 // Trim all trailing NULs and spaces.
 // The ustar and POSIX tar specs require a trailing NUL or
@@ -29,4 +37,9 @@
 end--;
 trailer = buffer[end - 1];
+}
+
+if (end <= start) {
+throw new IllegalArgumentException(
+exceptionMessage(buffer, offset, length, end, trailer));
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

---

## Compress-39

**Root cause:** The bug is that sanitize must cap the sanitized output to 255 characters and replace the final three characters with "..." when the input exceeds that limit, while still preserving the original character-filtering behavior for non-printable/special characters.

**Why pre-patch overfits:** The pre-patch fixes the length-limiting aspect for the tested long ASCII input, but it is semantically wrong because it stops treating SPECIALS-block characters as invalid and appends them directly. It would therefore produce incorrect sanitized output on reasonable inputs containing such characters, so it does not fully implement the method's intended behavior even though it passes the provided tests.

**OD refinement (NECESSARY):** The refinement restores the original UnicodeBlock-based filtering logic that the pre-patch removed, so characters from the SPECIALS block or with null UnicodeBlock are still converted to '?'. It also changes truncation detection from `len > maxLength` to `sb.length() >= maxLength`, which is needed when sanitization expands/replaces characters and the output reaches the limit independently of raw input length.

**Verdict:** Because the refinement change is NECESSARY, not cosmetic, the pre-patch was missing required sanitization logic and only happened to satisfy the observed test scenario. Therefore the original patch was an overfitting fix rather than a genuinely correct one.

**Before OD (overfitting):**
```diff
@@ -3,14 +3,16 @@
 final int len = chars.length;
 final StringBuilder sb = new StringBuilder();
-for (int i = 0; i < len; i++) {
+int maxLength = 255;
+for (int i = 0; i < len && sb.length() < maxLength; i++) {
 final char c = chars[i];
 if (!Character.isISOControl(c)) {
-Character.UnicodeBlock block = Character.UnicodeBlock.of(c);
-if (block != null && block != Character.UnicodeBlock.SPECIALS) {
 sb.append(c);
 continue;
 }
+sb.append('?');
 }
-sb.append('?');
+if (len > maxLength) {
+sb.setLength(maxLength - 3);
+sb.append("...");
 }
 return sb.toString();
```

**After OD (correct):**
```diff
@@ -1,7 +1,11 @@
 public static String sanitize(String s) {
+if (s == null) {
+return "";
+}
 final char[] chars = s.toCharArray();
 final int len = chars.length;
 final StringBuilder sb = new StringBuilder();
-for (int i = 0; i < len; i++) {
+int maxLength = 255;
+for (int i = 0; i < len && sb.length() < maxLength; i++) {
 final char c = chars[i];
 if (!Character.isISOControl(c)) {
@@ -14,4 +18,8 @@
 sb.append('?');
 }
+if (sb.length() >= maxLength) {
+sb.setLength(maxLength - 3);
+sb.append("...");
+}
 return sb.toString();
 }
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

## Compress-41

**Root cause:** The method handled central-directory signatures and invalid non-local-header signatures incorrectly by returning null for all non-LFH records instead of returning null only for end-of-archive markers and throwing ZipException for truly unexpected signatures.

**Why pre-patch overfits:** The pre-patch improves behavior for invalid signatures by throwing, but it does not implement the correct semantics: it continues past CFH/AED handling and only later returns null because those signatures are explicitly exempted from the exception check. That is not equivalent to the proper fix, and on reasonable inputs it can perform unnecessary or incorrect post-central-directory processing after skipRemainderOfArchive, so it is not a robust implementation of the root cause.

**OD refinement (NECESSARY):** The refinement changed the control flow so that CFH/AED signatures immediately return null after skipping the remainder, while every other non-LFH signature throws a ZipException. This is necessary because the pre-patch still treats CFH/AED as part of the generic invalid-signature check set, rather than separating valid end-of-archive markers from genuinely malformed records as the ground-truth fix does.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch was missing required control-flow logic rather than merely using a different equivalent structure. The refined patch matches the intended semantics of the ground-truth fix, while the pre-patch only passes tests with an incomplete handling of CFH/AED.

**Before OD (overfitting):**
```diff
@@ -11,8 +11,4 @@
 try {
 if (firstEntry) {
-// split archives have a special signature before the
-// first local file header - look for it and fail with
-// the appropriate error message if this is a split
-// archive.
 readFirstLocalFileHeader(LFH_BUF);
 } else {
@@ -27,4 +23,7 @@
 hitCentralDirectory = true;
 skipRemainderOfArchive();
+}
+if (!sig.equals(ZipLong.LFH_SIG) && !sig.equals(ZipLong.CFH_SIG) && !sig.equals(ZipLong.AED_SIG)) {
+throw new ZipException("Unexpected record signature: " + sig);
 }
 if (!sig.equals(ZipLong.LFH_SIG)) {
```

**After OD (correct):**
```diff
@@ -11,8 +11,4 @@
 try {
 if (firstEntry) {
-// split archives have a special signature before the
-// first local file header - look for it and fail with
-// the appropriate error message if this is a split
-// archive.
 readFirstLocalFileHeader(LFH_BUF);
 } else {
@@ -27,7 +23,8 @@
 hitCentralDirectory = true;
 skipRemainderOfArchive();
+return null;
 }
 if (!sig.equals(ZipLong.LFH_SIG)) {
-return null;
+throw new ZipException("Unexpected record signature: " + sig);
 }
 
```

**Ground truth fix:**
```diff
@@ -27,7 +27,8 @@
 hitCentralDirectory = true;
 skipRemainderOfArchive();
+return null;
 }
 if (!sig.equals(ZipLong.LFH_SIG)) {
-return null;
+throw new ZipException(String.format("Unexpected record signature: 0X%X", sig.getValue()));
 }
 
```

---

## Compress-43

**Root cause:** The bug is that data-descriptor usage was determined only from compression method/channel, so phased/raw entries could still be marked as using a data descriptor in the local header metadata/GPB even though their sizes and CRC were already known and no descriptor should be written.

**Why pre-patch overfits:** The pre-patch partially addresses the issue by making metadata and the local `dataDescriptor` boolean depend on `!phased`, matching the ground-truth direction. However, it leaves GPB encoding dependent on the internal behavior of `getGeneralPurposeBits(...)` and does not explicitly clear the DD bit, so it can still emit an incorrect GPB on reasonable inputs where the bit object retains prior state; thus it is not a robust semantic fix.

**OD refinement (NECESSARY):** The OD refinement added an explicit `generalPurposeBit.useDataDescriptor(dataDescriptor)` call before encoding the local header GPB, and also guarded `writeDataDescriptor` with `usesDataDescriptor(...)`. This is necessary because merely computing `dataDescriptor = usesDataDescriptor(zipMethod) && !phased` is not enough if `getGeneralPurposeBits(...)` returns a reused or preconfigured `GeneralPurposeBit` whose data-descriptor flag is still set; without explicitly clearing it, the LFH GPB can remain wrong for raw/phased entries.

**Verdict:** Because the refinement change is NECESSARY, not cosmetic, the pre-patch was missing logic required to reliably clear/synchronize the data-descriptor flag in the written header. Therefore the original patch was overfitting rather than already genuinely correct.

**Before OD (overfitting):**
```java
// --- L1024 ---
private void writeLocalFileHeader(final ZipArchiveEntry ze, final boolean phased) throws IOException {
    final boolean encodable = zipEncoding.canEncode(ze.getName());
    final ByteBuffer name = getName(ze);

    if (createUnicodeExtraFields != UnicodeExtraFieldPolicy.NEVER) {
        addUnicodeExtraFields(ze, encodable, name);
    }

    final long localHeaderStart = streamCompressor.getTotalBytesWritten();
    final byte[] localHeader = createLocalFileHeader(ze, name, encodable, phased, localHeaderStart);
    metaData.put(ze, new EntryMetaData(localHeaderStart, usesDataDescriptor(ze.getMethod())));
    entry.localDataStart = localHeaderStart + LFH_CRC_OFFSET; // At crc offset
    writeCounted(localHeader);
    entry.dataStart = streamCompressor.getTotalBytesWritten();
}

// --- L1041 ---
private byte[] createLocalFileHeader(final ZipArchiveEntry ze, final ByteBuffer name, final boolean encodable,
                                     final boolean phased, long archiveOffset) throws IOException {
    ResourceAlignmentExtraField oldAlignmentEx =
        (ResourceAlignmentExtraField) ze.getExtraField(ResourceAlignmentExtraField.ID);
    if (oldAlignmentEx != null) {
        ze.removeExtraField(ResourceAlignmentExtraField.ID);
    }

    int alignment = ze.getAlignment();
    if (alignment <= 0 && oldAlignmentEx != null) {
        alignment = oldAlignmentEx.getAlignment();
    }

    if (alignment > 1 || (oldAlignmentEx != null && !oldAlignmentEx.allowMethodChange())) {
        int oldLength = LFH_FILENAME_OFFSET +
                        name.limit() - name.position() +
                        ze.getLocalFileDataExtra().length;
// ... 91 more lines
```

**After OD (correct):**
```java
// --- L1024 ---
private void writeLocalFileHeader(final ZipArchiveEntry ze, final boolean phased) throws IOException {
    final boolean encodable = zipEncoding.canEncode(ze.getName());
    final ByteBuffer name = getName(ze);

    if (createUnicodeExtraFields != UnicodeExtraFieldPolicy.NEVER) {
        addUnicodeExtraFields(ze, encodable, name);
    }

    final long localHeaderStart = streamCompressor.getTotalBytesWritten();
    final byte[] localHeader = createLocalFileHeader(ze, name, encodable, phased, localHeaderStart);
    metaData.put(ze, new EntryMetaData(localHeaderStart, usesDataDescriptor(ze.getMethod()) && !phased));
    entry.localDataStart = localHeaderStart + LFH_CRC_OFFSET; // At crc offset
    writeCounted(localHeader);
    entry.dataStart = streamCompressor.getTotalBytesWritten();
}

// --- L1041 ---
private byte[] createLocalFileHeader(final ZipArchiveEntry ze, final ByteBuffer name, final boolean encodable,
                                     final boolean phased, long archiveOffset) throws IOException {
    ResourceAlignmentExtraField oldAlignmentEx =
        (ResourceAlignmentExtraField) ze.getExtraField(ResourceAlignmentExtraField.ID);
    if (oldAlignmentEx != null) {
        ze.removeExtraField(ResourceAlignmentExtraField.ID);
    }

    int alignment = ze.getAlignment();
    if (alignment <= 0 && oldAlignmentEx != null) {
        alignment = oldAlignmentEx.getAlignment();
    }

    if (alignment > 1 || (oldAlignmentEx != null && !oldAlignmentEx.allowMethodChange())) {
        int oldLength = LFH_FILENAME_OFFSET +
                        name.limit() - name.position() +
                        ze.getLocalFileDataExtra().length;
// ... 92 more lines
```

---

## Csv-1

**Root cause:** The bug is that line counting treats every '\n' as a new line but does not correctly support '\r' line endings and must avoid double-counting the '\n' in a CRLF sequence.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test with CR-only separators by counting '\r', but it does not fully fix the root cause because it also counts '\n' after '\r'. On reasonable inputs containing CRLF line endings, it would overcount line numbers, so it is not a generally correct implementation.

**OD refinement (NECESSARY):** The refinement added state-dependent logic using lastChar so that '\r' increments the line counter, but a '\n' only increments it when it is not immediately preceded by '\r'. This is necessary because the pre-patch counts both characters in a CRLF pair as two line breaks, which is semantically wrong for standard line-ending handling.

**Verdict:** Because the refinement change is NECESSARY and introduces missing logic to prevent double-counting CRLF, the pre-patch was an overfitting fix rather than a semantically correct one.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,5 @@
 public int read() throws IOException {
 int current = super.read();
-if (current == '\n') {
+if (current == '\n' || current == '\r') {
 lineCounter++;
 }
```

**After OD (correct):**
```diff
@@ -2,4 +2,8 @@
 int current = super.read();
 if (current == '\n') {
+if (lastChar != '\r') {
+lineCounter++;
+}
+} else if (current == '\r') {
 lineCounter++;
 }
```

**Ground truth fix:**
```diff
@@ -1,5 +1,5 @@
 public int read() throws IOException {
 int current = super.read();
-if (current == '\n') {
+if (current == '\r' || (current == '\n' && lastChar != '\r')) {
 lineCounter++;
 }
```

---

## Csv-2

### Patch 1

**Root cause:** The bug occurs because get(String) directly indexes into values using a header-mapped index without validating that the index is within the record's bounds, so inconsistent header/record sizes cause an ArrayIndexOutOfBoundsException instead of the intended behavior.

**Why pre-patch overfits:** The pre-patch fixes the specific out-of-bounds case from the failing test, but it does so by collapsing two distinct cases: missing header names and inconsistent record indices. For any reasonable input where the requested name is not present in the mapping, the buggy original and ground-truth behavior is to return null, whereas the pre-patch throws IllegalArgumentException, so it is semantically wrong beyond the tested case.

**OD refinement (NECESSARY):** The refinement restored the original null-return behavior for unknown header names by explicitly returning null when mapping.get(name) is null, and separately checked bounds only for non-null indices. This is necessary because the pre-patch incorrectly throws IllegalArgumentException for missing headers, changing valid existing semantics rather than only fixing inconsistent-record handling.

**Verdict:** Because the OD refinement made a NECESSARY semantic correction?봯reserving null for absent headers while only rejecting out-of-range mapped indices?봳he pre-patch was overfitting to the failing test rather than fully implementing the correct behavior.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,8 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("The record is inconsistent.");
 }
+}
```

**After OD (correct):**
```diff
@@ -5,4 +5,10 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index < 0 || index >= values.length) {
+throw new IllegalArgumentException("Inconsistent record: Index out of bounds");
+}
+return values[index.intValue()];
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

**Root cause:** The method accesses values[index] without handling the case where a header maps to an index outside the record's value array, so inconsistent records throw ArrayIndexOutOfBoundsException instead of IllegalArgumentException.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test by rejecting out-of-range indices, but it also throws IllegalArgumentException whenever mapping.get(name) returns null. That is semantically wrong for reasonable inputs where the requested header name is simply absent, because the original and ground-truth behavior is to return null in that case.

**OD refinement (NECESSARY):** The refinement changed the behavior for missing headers (index == null) from throwing IllegalArgumentException to returning null, while still throwing IllegalArgumentException only for out-of-bounds mapped indices. This is necessary because the original contract and ground-truth behavior distinguish between an unknown header name and an inconsistent record; the pre-patch incorrectly conflates them.

**Verdict:** Because the OD refinement introduced necessary missing logic for the index == null case, the pre-patch was not semantically correct. It overfits by passing the observed test while breaking valid behavior for unknown header names.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,8 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index >= 0 && index < values.length) {
+return values[index.intValue()];
+} else {
+throw new IllegalArgumentException("Inconsistent record: index out of bounds");
 }
+}
```

**After OD (correct):**
```diff
@@ -5,4 +5,10 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index < 0 || index >= values.length) {
+throw new IllegalArgumentException("Inconsistent record: Index out of bounds");
+}
+return values[index.intValue()];
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

**Root cause:** The bug occurs because get(String) directly indexes into values using the header mapping without validating that the mapped index is within the record's bounds, so inconsistent header/record sizes cause an ArrayIndexOutOfBoundsException instead of the intended behavior.

**Why pre-patch overfits:** The pre-patch fixes the specific out-of-bounds case from the failing test, but it does so by collapsing two distinct cases: missing header names and inconsistent records. For any reasonable input where the requested name is not present in the header mapping, the buggy/original behavior should return null, whereas the pre-patch throws IllegalArgumentException, so it is semantically wrong beyond the tested case.

**OD refinement (NECESSARY):** The refinement restored the original null semantics for unknown header names by returning null when mapping.get(name) is null, and it also added an explicit lower-bound check for negative indices. This is necessary because the pre-patch incorrectly throws IllegalArgumentException for missing headers, changing existing behavior unrelated to inconsistent records.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic that preserves correct behavior for index == null, the pre-patch was overfitting to the failing test rather than fully fixing the method semantics.

**Before OD (overfitting):**
```diff
@@ -5,4 +5,4 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index != null && index < values.length) { return values[index.intValue()]; } else { throw new IllegalArgumentException("Record is inconsistent"); }
 }
```

**After OD (correct):**
```diff
@@ -5,4 +5,10 @@
 }
 final Integer index = mapping.get(name);
-return index != null ? values[index.intValue()] : null;
+if (index == null) {
+return null;
 }
+if (index < 0 || index >= values.length) {
+throw new IllegalArgumentException("Inconsistent record: Index out of bounds");
+}
+return values[index.intValue()];
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

## JacksonCore-8

### Patch 1

**Root cause:** getTextBuffer() returns _currentSegment whenever !_hasSegments, but after resetWithEmpty() _currentSegment can be null, causing a NullPointerException instead of returning an empty char array.

**Why pre-patch overfits:** The pre-patch does not correctly implement the method's intended behavior: for any non-segmented buffer with actual text in _currentSegment, it returns new char[0] rather than the stored characters. So while it fixes the tested null case, it would produce incorrect results on reasonable inputs outside the failing test, making it an overfitting patch.

**OD refinement (NECESSARY):** The refinement changed the behavior for the case !_hasSegments && _currentSegment != null: instead of always returning a new empty array whenever !_hasSegments, it returns the existing _currentSegment and only substitutes an empty array when _currentSegment is null. This is necessary because the pre-patch discards valid buffered content in the common single-segment case and changes method semantics beyond the empty-buffer bug.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic rewrite, and the pre-patch would incorrectly return an empty array for valid non-segmented content, the original patch was a true overfitting fix.

**Before OD (overfitting):**
```diff
@@ -8,5 +8,5 @@
 }
 // Nope; but does it fit in just one segment?
-if (!_hasSegments)  return _currentSegment;
+if (!_hasSegments || _currentSegment == null)  return new char[0];
 // Nope, need to have/create a non-segmented array and return it
 return contentsAsArray();
```

**After OD (correct):**
```diff
@@ -8,5 +8,11 @@
 }
 // Nope; but does it fit in just one segment?
-if (!_hasSegments)  return _currentSegment;
+if (!_hasSegments) {
+if (_currentSegment == null) {
+// Handle the case where _currentSegment is null
+_currentSegment = new char[0]; // or initialize with appropriate size
+}
+return _currentSegment;
+}
 // Nope, need to have/create a non-segmented array and return it
 return contentsAsArray();
```

**Ground truth fix:**
```diff
@@ -8,5 +8,5 @@
 }
 // Nope; but does it fit in just one segment?
-if (!_hasSegments)  return _currentSegment;
+if (!_hasSegments && _currentSegment != null)  return _currentSegment;
 // Nope, need to have/create a non-segmented array and return it
 return contentsAsArray();
```

### Patch 2

**Root cause:** The bug is that getTextBuffer() returns _currentSegment when !_hasSegments even if _currentSegment is null, instead of falling back to contentsAsArray() for the empty-buffer case.

**Why pre-patch overfits:** The pre-patch does not fix the real root cause; it misdiagnoses the issue as _resultString being null and returns a hardcoded empty array for all such states. That means it will incorrectly return empty content for many reasonable non-empty TextBuffer states where _resultString is null but characters are available in _currentSegment or segments, so it is overfitting to the failing empty-buffer test.

**OD refinement (NECESSARY):** The refinement moved the empty-array fallback so it only applies when no valid backing source exists, and it restored the key condition that _currentSegment must be non-null before returning it. This is necessary because the pre-patch unconditionally returns a new empty array whenever _resultString is null, which bypasses valid data stored in _currentSegment or segmented contents.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic the pre-patch lacks, the original patch is a true overfitting fix. It passes the observed test by coincidence but is semantically wrong for broader valid inputs.

**Before OD (overfitting):**
```diff
@@ -7,4 +7,8 @@
 return (_resultArray = _resultString.toCharArray());
 }
+// Handle the case where _resultString is null
+if (_resultString == null) {
+return new char[0];
+}
 // Nope; but does it fit in just one segment?
 if (!_hasSegments)  return _currentSegment;
```

**After OD (correct):**
```diff
@@ -2,5 +2,5 @@
 {
 // Are we just using shared input buffer?
-if (_inputStart >= 0) return _inputBuffer;
+if (_inputStart >= 0 && _inputBuffer != null) return _inputBuffer;
 if (_resultArray != null)  return _resultArray;
 if (_resultString != null) {
@@ -8,5 +8,9 @@
 }
 // Nope; but does it fit in just one segment?
-if (!_hasSegments)  return _currentSegment;
+if (!_hasSegments && _currentSegment != null)  return _currentSegment;
+// Handle cases where all relevant fields are null
+if (_inputBuffer == null && _resultArray == null && _resultString == null && (!_hasSegments || _currentSegment == null)) {
+return new char[0];
+}
 // Nope, need to have/create a non-segmented array and return it
 return contentsAsArray();
```

**Ground truth fix:**
```diff
@@ -8,5 +8,5 @@
 }
 // Nope; but does it fit in just one segment?
-if (!_hasSegments)  return _currentSegment;
+if (!_hasSegments && _currentSegment != null)  return _currentSegment;
 // Nope, need to have/create a non-segmented array and return it
 return contentsAsArray();
```

---

## JacksonDatabind-108

### Patch 1

**Root cause:** The bug is that ObjectReader.readTree(JsonParser) should return null when the parser has no content at EOF, but the buggy implementation always delegates to _bindAsTree, which yields a MissingNode instead.

**Why pre-patch overfits:** The pre-patch fixes the empty-input case only when the parser is initially unpositioned, matching the failing tests, but it does not correctly preserve parser state for general callers. If readTree(p) is invoked when p.currentToken() is already set to START_OBJECT, START_ARRAY, VALUE_NULL, etc., the extra nextToken() advances past the first token and changes semantics, so it is an overfitting fix rather than a correct implementation of the root cause.

**OD refinement (NECESSARY):** The refinement changed the method from unconditionally calling p.nextToken() to first checking p.currentToken() and only advancing when the parser has not yet been positioned. This is necessary because the pre-patch consumes one token even when the caller has already advanced the parser, which can skip the actual first token and misread or fail on valid non-empty inputs.

**Verdict:** Because the refinement introduced necessary parser-state-aware logic that the pre-patch lacked, the original patch is not semantically correct. It passes the observed tests by coincidence on uninitialized parsers but breaks reasonable cases with already-advanced parsers.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,4 @@
-@SuppressWarnings("unchecked")
 @Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
-return (T) _bindAsTree(p);
+if (p.nextToken() == null) { return null; } else { return (T) _bindAsTree(p); }
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
-@SuppressWarnings("unchecked")
-@Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
+JsonToken currentToken = p.currentToken();
+if (currentToken == null && p.nextToken() == null) {
+return null;
+}
 return (T) _bindAsTree(p);
 }
```

**Ground truth fix:**
```diff
@@ -2,4 +2,4 @@
 @Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
-return (T) _bindAsTree(p);
+return (T) _bindAsTreeOrNull(p);
 }
```

### Patch 2

**Root cause:** ObjectReader.readTree(JsonParser) should return null when the parser has no content/EOF, but the buggy code always delegates to _bindAsTree, which yields a MissingNode instead.

**Why pre-patch overfits:** The pre-patch fixes the empty-input case only when the parser is initially unpositioned, matching the failing tests, but it does not correctly preserve parser state for general readTree(JsonParser) usage. On reasonable inputs where the parser is already at START_OBJECT/START_ARRAY/value before readTree is called, the extra nextToken() advances past the first token and changes semantics, so it is an overfitting fix rather than a correct implementation of the root cause.

**OD refinement (NECESSARY):** The refinement changed the method from unconditionally advancing the parser with p.nextToken() to first checking p.currentToken() and only advancing when the parser has not yet been positioned. This is necessary because the pre-patch consumes one token even when the caller has already advanced the parser, which can skip the actual first token and misread or fail on valid inputs.

**Verdict:** Because the OD refinement added necessary state-sensitive logic to avoid incorrectly advancing an already-positioned parser, the pre-patch was not semantically correct. Its behavior is wrong beyond the specific EOF tests, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,3 @@
-@SuppressWarnings("unchecked")
-@Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
-return (T) _bindAsTree(p);
+if (p.nextToken() == null) { return null; } else { return (T) _bindAsTree(p); }
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
-@SuppressWarnings("unchecked")
-@Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
+JsonToken currentToken = p.currentToken();
+if (currentToken == null && p.nextToken() == null) {
+return null;
+}
 return (T) _bindAsTree(p);
 }
```

**Ground truth fix:**
```diff
@@ -2,4 +2,4 @@
 @Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
-return (T) _bindAsTree(p);
+return (T) _bindAsTreeOrNull(p);
 }
```

### Patch 3

**Root cause:** ObjectReader.readTree(JsonParser) should return null only when the parser is truly at end-of-input, but the buggy code delegates to _bindAsTree which yields a MissingNode for empty input instead.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause because it returns null whenever the parser has no current token, even if the input has unread content and the parser simply has not been advanced yet. That means it would incorrectly return null for reasonable uses like calling `reader().readTree(p)` on a newly created parser over non-empty JSON, so it overfits to the empty-input tests.

**OD refinement (NECESSARY):** The refinement changed the condition from `!p.hasCurrentToken()` to `currentToken == null && p.nextToken() == null`, which distinguishes between an uninitialized parser and an actually empty/EOF parser. This is necessary because a fresh parser commonly has no current token before reading begins, yet may still contain valid JSON that should be parsed rather than immediately returning null.

**Verdict:** Because the refinement added necessary logic to advance/check the parser before deciding EOF, the pre-patch was semantically wrong rather than an alternative correct fix. Its `hasCurrentToken()` check is too broad and causes false null results on valid non-empty inputs.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,4 @@
-@SuppressWarnings("unchecked")
-@Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
+if (!p.hasCurrentToken()) { return null; }
 return (T) _bindAsTree(p);
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
-@SuppressWarnings("unchecked")
-@Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
+JsonToken currentToken = p.currentToken();
+if (currentToken == null && p.nextToken() == null) {
+return null;
+}
 return (T) _bindAsTree(p);
 }
```

**Ground truth fix:**
```diff
@@ -2,4 +2,4 @@
 @Override
 public <T extends TreeNode> T readTree(JsonParser p) throws IOException {
-return (T) _bindAsTree(p);
+return (T) _bindAsTreeOrNull(p);
 }
```

---

## JacksonDatabind-12

### Patch 1

**Root cause:** MapDeserializer was incorrectly considered cacheable even when it had property-specific custom key or value deserializers, causing a cached instance with one property's custom deserializer to be reused for another property.

**Why pre-patch overfits:** The pre-patch only prevents caching when `_valueDeserializer` is present, which fixes the observed failing test for custom value deserializers but leaves the analogous custom-key-deserializer case broken. On reasonable inputs involving annotated custom key deserializers on one map property but not another, the deserializer could still be wrongly cached and reused, so the root cause is only partially fixed.

**OD refinement (NECESSARY):** The refinement added the missing `_keyDeserializer == null` condition to `isCachable()`, so maps with custom key deserializers are also excluded from caching. This is required by the method comment and the ground-truth fix, because custom key deserializers create the same kind of property-specific state as custom value deserializers.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic or equivalent change, the pre-patch was an incomplete fix. It overfits to the tested custom-value-deserializer scenario and does not fully resolve the caching bug class.

**Before OD (overfitting):**
```diff
@@ -4,4 +4,5 @@
 */
 return (_valueTypeDeserializer == null)
+&& (_valueDeserializer == null)
 && (_ignorableProperties == null);
 }
```

**After OD (correct):**
```diff
@@ -4,4 +4,6 @@
 */
 return (_valueTypeDeserializer == null)
-&& (_ignorableProperties == null);
+&& (_ignorableProperties == null)
+&& (_valueDeserializer == null)
+&& (_keyDeserializer == null);
 }
```

**Ground truth fix:**
```diff
@@ -3,5 +3,7 @@
 * if annotated to use non-standard one) should also prevent caching.
 */
-return (_valueTypeDeserializer == null)
+return (_valueDeserializer == null)
+&& (_keyDeserializer == null)
+&& (_valueTypeDeserializer == null)
 && (_ignorableProperties == null);
 }
```

### Patch 2

**Root cause:** MapDeserializer.isCachable() incorrectly allowed caching even when a custom key or value deserializer was present, causing annotation-specific deserializers to be reused for other map properties.

**Why pre-patch overfits:** The pre-patch fixes the observed failing test by preventing caching when `_valueDeserializer` is present, but it does not fully fix the root cause because it still allows caching when only a custom key deserializer is present. On reasonable inputs involving annotated custom map key deserializers on one property but not another, it could still incorrectly reuse a cached deserializer and produce wrong behavior.

**OD refinement (NECESSARY):** The OD refinement added a missing check for `_keyDeserializer == null`, so caching is also disabled when a custom key deserializer is configured. This is semantically important because the method comment and ground-truth fix both require that either custom value or custom key deserializers prevent caching; the pre-patch only handled the value-deserializer case.

**Verdict:** Because the refinement added missing runtime logic required by the actual specification of cachability, the change is NECESSARY rather than cosmetic. Therefore the pre-patch is an incomplete, overfitting fix that passes current tests but remains semantically wrong for custom-key-deserializer cases.

**Before OD (overfitting):**
```diff
@@ -4,4 +4,4 @@
 */
 return (_valueTypeDeserializer == null)
-&& (_ignorableProperties == null);
+&& (_ignorableProperties == null) && (_valueDeserializer == null);
 }
```

**After OD (correct):**
```diff
@@ -4,4 +4,6 @@
 */
 return (_valueTypeDeserializer == null)
-&& (_ignorableProperties == null);
+&& (_ignorableProperties == null)
+&& (_valueDeserializer == null)
+&& (_keyDeserializer == null);
 }
```

**Ground truth fix:**
```diff
@@ -3,5 +3,7 @@
 * if annotated to use non-standard one) should also prevent caching.
 */
-return (_valueTypeDeserializer == null)
+return (_valueDeserializer == null)
+&& (_keyDeserializer == null)
+&& (_valueTypeDeserializer == null)
 && (_ignorableProperties == null);
 }
```

### Patch 3

**Root cause:** MapDeserializer was considered cacheable even when it had custom key/value deserializers or ignorable properties, causing a context-specific deserializer instance to be incorrectly reused for other map properties.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test by preventing caching when `_valueDeserializer` is present, but it does not fully fix the root cause. It would still incorrectly cache deserializers when a custom key deserializer is present, and it could also cache despite non-null ignorable properties because that check was removed, so it is semantically incomplete beyond the observed test.

**OD refinement (NECESSARY):** The refinement restored the original `_ignorableProperties == null` condition and added `_keyDeserializer == null`, so caching is now disabled for all non-standard, property-specific map deserializer configurations. This is necessary because the pre-patch only handled custom value deserializers and also accidentally dropped the ignorable-properties safeguard, leaving other context-sensitive cases incorrectly cacheable.

**Verdict:** Because the refinement added missing runtime conditions that are required for correct cacheability semantics, the change is NECESSARY rather than cosmetic. Therefore the pre-patch is a true overfitting patch: it addresses one tested manifestation but not the full bug.

**Before OD (overfitting):**
```diff
@@ -3,5 +3,4 @@
 * if annotated to use non-standard one) should also prevent caching.
 */
-return (_valueTypeDeserializer == null)
-&& (_ignorableProperties == null);
+return (_valueTypeDeserializer == null) && (_valueDeserializer == null);
 }
```

**After OD (correct):**
```diff
@@ -4,4 +4,6 @@
 */
 return (_valueTypeDeserializer == null)
-&& (_ignorableProperties == null);
+&& (_ignorableProperties == null)
+&& (_valueDeserializer == null)
+&& (_keyDeserializer == null);
 }
```

**Ground truth fix:**
```diff
@@ -3,5 +3,7 @@
 * if annotated to use non-standard one) should also prevent caching.
 */
-return (_valueTypeDeserializer == null)
+return (_valueDeserializer == null)
+&& (_keyDeserializer == null)
+&& (_valueTypeDeserializer == null)
 && (_ignorableProperties == null);
 }
```

---

## JacksonDatabind-19

**Root cause:** TypeFactory._mapType incorrectly treats java.util.Properties like a generic raw Map, inferring Object/Object instead of the special String/String key and value types that Properties semantically uses.

**Why pre-patch overfits:** The pre-patch does fix the failing Properties tests, but it does so by replacing all map type inference with a constant String/String pair regardless of rawClass. That is semantically wrong for reasonable inputs such as HashMap<Integer,Long>, Map<String,Integer>, or raw non-Properties map classes, which would all be mis-typed as Map<String,String>.

**OD refinement (NECESSARY):** The OD refinement changed the method from always hardcoding String/String for every map-like class to applying that special handling only when the raw class is Properties (or a subclass), and otherwise preserving normal generic type inference via findTypeParameters. This is necessary because _mapType is used for all Map types, not just Properties, so the pre-patch breaks ordinary maps by forcing incorrect String/String types.

**Verdict:** Because the refinement introduced necessary missing logic?봱estricting the String/String special case to Properties and restoring generic handling for other maps?봳he pre-patch was a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -2,5 +2,5 @@
 {
 // 28-May-2015, tatu: Properties are special, as per [databind#810]
-JavaType[] typeParams = findTypeParameters(rawClass, Map.class);
+JavaType[] typeParams = new JavaType[] { _constructType(String.class, null), _constructType(String.class, null) };
 // ok to have no types ("raw")
 if (typeParams == null) {
```

**After OD (correct):**
```diff
@@ -2,13 +2,25 @@
 {
 // 28-May-2015, tatu: Properties are special, as per [databind#810]
+JavaType keyType = _unknownType();
+JavaType valueType = _unknownType();
+
+// Check if the class is Properties
+if (Properties.class.isAssignableFrom(rawClass)) {
+// Assuming that Properties are always String -> String
+keyType = _constructType(String.class, null);
+valueType = _constructType(String.class, null);
+} else {
 JavaType[] typeParams = findTypeParameters(rawClass, Map.class);
 // ok to have no types ("raw")
-if (typeParams == null) {
-return MapType.construct(rawClass, _unknownType(), _unknownType());
+if (typeParams == null || typeParams.length == 0) {
+return MapType.construct(rawClass, keyType, valueType);
 }
 // but exactly 2 types if any found
 if (typeParams.length != 2) {
-throw new IllegalArgumentException("Strange Map type "+rawClass.getName()+": can not determine type parameters");
+throw new IllegalArgumentException("Strange Map type " + rawClass.getName() + ": can not determine type parameters");
 }
-return MapType.construct(rawClass, typeParams[0], typeParams[1]);
+keyType = typeParams[0];
+valueType = typeParams[1];
 }
+return MapType.construct(rawClass, keyType, valueType);
+}
```

**Ground truth fix:**
```diff
@@ -2,4 +2,7 @@
 {
 // 28-May-2015, tatu: Properties are special, as per [databind#810]
+if (rawClass == Properties.class) {
+return MapType.construct(rawClass, CORE_TYPE_STRING, CORE_TYPE_STRING);
+}
 JavaType[] typeParams = findTypeParameters(rawClass, Map.class);
 // ok to have no types ("raw")
```

---

## JacksonDatabind-71

**Root cause:** The bug is that StdKeyDeserializer.forType does not recognize CharSequence key types as string-like map keys, so no key deserializer is returned for CharSequence.

**Why pre-patch overfits:** The pre-patch fixes the immediate failing test for Map<CharSequence, String> by adding a special case for exactly CharSequence.class, but it does not fully address the underlying issue for other reasonable key types that implement CharSequence, such as StringBuilder or StringBuffer. Those types would still return null and fail key deserializer lookup, so the patch is too narrow and overfits to the observed test input.

**OD refinement (NECESSARY):** The refinement changed the added condition from an exact match on raw == CharSequence.class to a generalized check CharSequence.class.isAssignableFrom(raw). This broadens support from only the CharSequence interface itself to all CharSequence implementations, which is required to correctly treat string-like key types uniformly rather than only satisfying the specific failing test.

**Verdict:** Because the OD refinement introduced necessary missing generalization logic, the pre-patch was not semantically complete. It handled only the exact tested type instead of the full class of CharSequence-based key types, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -41,4 +41,6 @@
 FromStringDeserializer<?> deser = FromStringDeserializer.findDeserializer(Currency.class);
 return new StdKeyDeserializer(TYPE_CURRENCY, raw, deser);
+} else if (raw == CharSequence.class) {
+return StringKD.forType(raw);
 } else {
 return null;
```

**After OD (correct):**
```diff
@@ -41,4 +41,6 @@
 FromStringDeserializer<?> deser = FromStringDeserializer.findDeserializer(Currency.class);
 return new StdKeyDeserializer(TYPE_CURRENCY, raw, deser);
+} else if (CharSequence.class.isAssignableFrom(raw)) {
+return StringKD.forType(raw);
 } else {
 return null;
```

**Ground truth fix:**
```diff
@@ -4,5 +4,5 @@
 
 // first common types:
-if (raw == String.class || raw == Object.class) {
+if (raw == String.class || raw == Object.class || raw == CharSequence.class) {
 return StringKD.forType(raw);
 } else if (raw == UUID.class) {
```

---

## JacksonDatabind-91

**Root cause:** _hasCustomHandlers failed to consider custom key deserializers for map-like container types, causing deserializers with custom key handlers to be incorrectly cached and reused.

**Why pre-patch overfits:** The pre-patch fixes the observed test by detecting a custom key handler on map types, but it does so in the wrong place and without restricting the check to map-like types. On reasonable other inputs such as collection/array-like container types with a content type but no key type, it can misbehave or throw due to dereferencing `t.getKeyType()` when no key type exists, so it is not a generally correct fix.

**OD refinement (NECESSARY):** The refinement moved the key-handler check out of the `ct != null` early-return path and made it a separate check after content-type handling. This is necessary because the pre-patch unconditionally calls `t.getKeyType().getValueHandler()` for every container with non-null content type, which can be invalid for non-map containers and also omits the intended map-like guard present in the ground-truth fix.

**Verdict:** Because the OD refinement introduced necessary logic rather than a cosmetic refactor, the pre-patch was not semantically correct. It passed the tests by coincidence for the exercised map case but remained wrong for other container types.

**Before OD (overfitting):**
```diff
@@ -4,7 +4,7 @@
 JavaType ct = t.getContentType();
 if (ct != null) {
-return (ct.getValueHandler() != null) || (ct.getTypeHandler() != null);
+return (t.getKeyType().getValueHandler() != null) || (ct.getValueHandler() != null) || (ct.getTypeHandler() != null);
+}
 // Second: map(-like) types may have value handler for key (but not type; keys are untyped)
-}
 }
 return false;
```

**After OD (correct):**
```diff
@@ -4,6 +4,12 @@
 JavaType ct = t.getContentType();
 if (ct != null) {
-return (ct.getValueHandler() != null) || (ct.getTypeHandler() != null);
+if ((ct.getValueHandler() != null) || (ct.getTypeHandler() != null)) {
+return true;
+}
+}
 // Second: map(-like) types may have value handler for key (but not type; keys are untyped)
+JavaType kt = t.getKeyType();
+if (kt != null && kt.getValueHandler() != null) {
+return true;
 }
 }
```

**Ground truth fix:**
```diff
@@ -4,6 +4,14 @@
 JavaType ct = t.getContentType();
 if (ct != null) {
-return (ct.getValueHandler() != null) || (ct.getTypeHandler() != null);
+if ((ct.getValueHandler() != null) || (ct.getTypeHandler() != null)) {
+return true;
+}
+}
 // Second: map(-like) types may have value handler for key (but not type; keys are untyped)
+if (t.isMapLikeType()) {
+JavaType kt = t.getKeyType();
+if (kt.getValueHandler() != null) {
+return true;
+}
 }
 }
```

---

## Jsoup-53

**Root cause:** The bug is that chompBalanced counts open/close delimiter characters even when they appear inside quoted attribute values, so selectors like [data='End]'] are parsed as prematurely closed.

**Why pre-patch overfits:** The pre-patch does fix the main root cause for the actual selector syntax exercised by jsoup here: quoted attribute values using single or double quotes, including escaped quotes. However, it remains incomplete relative to the refined behavior because it mishandles reasonable inputs containing backtick-quoted sections with delimiters inside them, where it would still incorrectly adjust depth.

**OD refinement (NECESSARY):** The refinement changes quote detection from only single and double quotes to also treating backticks as quote delimiters. This is a real semantic change: the pre-patch will toggle inQuote only for ' and ", while the refined patch also ignores balanced delimiters inside backtick-quoted sections.

**Verdict:** Because the refinement introduced necessary missing logic rather than a cosmetic change, and the pre-patch lacks that behavior, the original patch was not fully semantically correct. It passed the tests but still overfits by not handling all quote forms the refined correct patch supports.

**Before OD (overfitting):**
```diff
@@ -4,15 +4,21 @@
 int depth = 0;
 char last = 0;
+boolean inQuote = false;
+final char QUOTE1 = '\'';
+final char QUOTE2 = '"';
+final char ESC = '\\';
 
 do {
 if (isEmpty()) break;
 Character c = consume();
+if (last != ESC && (c == QUOTE1 || c == QUOTE2))
+inQuote = !inQuote;
 if (last == 0 || last != ESC) {
-if (c.equals(open)) {
+if (c == open && !inQuote) {
 depth++;
 if (start == -1)
 start = pos;
 }
-else if (c.equals(close))
+else if (c == close && !inQuote)
 depth--;
 }
```

**After OD (correct):**
```diff
@@ -4,15 +4,19 @@
 int depth = 0;
 char last = 0;
+boolean inQuote = false;
+final char ESC = '\\';
 
 do {
 if (isEmpty()) break;
 Character c = consume();
+if (last != ESC && (c == '"' || c == '\'' || c == '`'))
+inQuote = !inQuote;
 if (last == 0 || last != ESC) {
-if (c.equals(open)) {
+if (c == open && !inQuote) {
 depth++;
 if (start == -1)
 start = pos;
 }
-else if (c.equals(close))
+else if (c == close && !inQuote)
 depth--;
 }
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

## Lang-28

**Root cause:** The bug is that numeric entities for supplementary Unicode code points are written as a single UTF-16 code unit via Writer.write(int) instead of being converted to the required surrogate pair.

**Why pre-patch overfits:** The pre-patch does fix the main supplementary-character root cause by using Character.toChars(entityValue), matching the ground-truth fix for valid entities. However, it is not fully correct semantically because it will still throw on reasonable malformed inputs such as unterminated entities, short inputs, or values above Character.MAX_CODE_POINT, so it is not merely an alternative equivalent implementation.

**OD refinement (NECESSARY):** The OD refinement added bounds checks, validation for missing semicolons and empty entities, and code-point range checks before calling Character.toChars(entityValue). These are real semantic changes: the pre-patch can throw exceptions or mis-handle invalid numeric entities, whereas the refined patch safely returns 0 for such malformed or out-of-range inputs.

**Verdict:** Because the refinement introduced necessary missing validation logic, the pre-patch was not semantically robust and could fail outside the observed tests. Therefore it was an overfitting patch rather than a genuinely correct fix.

**Before OD (overfitting):**
```diff
@@ -27,5 +27,5 @@
 }
 
-out.write(entityValue);
+out.write(Character.toChars(entityValue));
 return 2 + (end - start) + (isHex ? 1 : 0) + 1;
 }
```

**After OD (correct):**
```diff
@@ -1,10 +1,16 @@
 public int translate(CharSequence input, int index, Writer out) throws IOException {
-// TODO: Protect from ArrayIndexOutOfBounds
-if(input.charAt(index) == '&' && input.charAt(index + 1) == '#') {
+if (input == null || index < 0 || index >= input.length() - 1) {
+return 0;
+}
+
+if (input.charAt(index) == '&' && input.charAt(index + 1) == '#') {
 int start = index + 2;
+if (start >= input.length()) {
+return 0;
+}
+
 boolean isHex = false;
-
 char firstChar = input.charAt(start);
-if(firstChar == 'x' || firstChar == 'X') {
+if (firstChar == 'x' || firstChar == 'X') {
 start++;
 isHex = true;
@@ -12,20 +18,24 @@
 
 int end = start;
-while(input.charAt(end) != ';') {
+while (end < input.length() && input.charAt(end) != ';') {
 end++;
 }
 
-int entityValue;
-try {
-if(isHex) {
-entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 16);
-} else {
-entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 10);
-}
-} catch(NumberFormatException nfe) {
+if (end >= input.length() || end == start) {
 return 0;
 }
 
-out.write(entityValue);
+String numStr = input.subSequence(start, end).toString();
+int entityValue;
+try {
+entityValue = isHex ? Integer.parseInt(numStr, 16) : Integer.parseInt(numStr, 10);
+if (entityValue < 0 || entityValue > Character.MAX_CODE_POINT) {
+return 0;
+}
+} catch (NumberFormatException nfe) {
+return 0;
+}
+
+out.write(Character.toChars(entityValue));
 return 2 + (end - start) + (isHex ? 1 : 0) + 1;
 }
```

**Ground truth fix:**
```diff
@@ -27,5 +27,11 @@
 }
 
+if(entityValue > 0xFFFF) {
+char[] chrs = Character.toChars(entityValue);
+out.write(chrs[0]);
+out.write(chrs[1]);
+} else {
 out.write(entityValue);
+}
 return 2 + (end - start) + (isHex ? 1 : 0) + 1;
 }
```

---

## Math-15

### Patch 1

**Root cause:** The bug is that for negative bases the code treats exponents with magnitude at or above 2^52 as definitely even integers, but doubles only guarantee integer/parity properties at or above 2^53, so some odd integers near 5e15 are misclassified.

**Why pre-patch overfits:** The pre-patch does make the failing test pass, but it does so by broadening the condition to 'almost integer' exponents, which is not mathematically correct for `pow` with a negative base. For reasonable inputs like `x = -2` and `y = 3.000000000000001`, it would incorrectly return approximately `-8` instead of `NaN`, so it overfits by using an ad hoc tolerance rather than fixing the actual 2^52 vs 2^53 threshold issue.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch's tolerance-based check `Math.abs(y - Math.round(y)) < 1e-14` and restored exact integer detection with `y == (long) y` for negative bases. This is a real semantic change: the pre-patch accepts non-integer exponents that are merely very close to an integer and assigns them a sign based on rounding, whereas the refined patch correctly returns NaN for such cases and only applies parity logic to true integer exponents.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch is not semantically equivalent to the confirmed-correct version. It passes the observed test by coincidence through an incorrect near-integer heuristic and would misbehave on other valid inputs.

**Before OD (overfitting):**
```diff
@@ -99,11 +99,7 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
-}
-
-if (y == (long) y) {
-// If y is an integer
-return ((long)y & 1) == 0 ? pow(-x, y) : -pow(-x, y);
+if (Math.abs(y - Math.round(y)) < 1e-14) {
+long roundedY = Math.round(y);
+return ((roundedY & 1) == 0) ? pow(-x, y) : -pow(-x, y);
 } else {
 return Double.NaN;
```

**After OD (correct):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -99,8 +98,4 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
-}
-
 if (y == (long) y) {
 // If y is an integer
@@ -155,5 +150,4 @@
 
 final double result = exp(lna, z, null);
-//result = result + result * z;
 return result;
 }
```

**Ground truth fix:**
```diff
@@ -99,5 +99,5 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+if (y >= TWO_POWER_53 || y <= -TWO_POWER_53) {
 return pow(-x, y);
 }
```

### Patch 2

**Root cause:** The bug is that for negative bases, the code assumes any exponent with magnitude at least 2^52 is an even integer, but doubles can still represent odd integers exactly up to 2^53, so parity is misclassified for values like 5e15 + 1.

**Why pre-patch overfits:** The pre-patch fixes the observed test for x = -1 by detecting odd/even large exponents, but it does so with semantically wrong logic for every other negative base. For example, inputs like pow(-2, 2^52) or pow(-2, 2^52 + 2) would incorrectly return 1.0 instead of a huge positive value, so it does not correctly fix the root cause in general.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch's special-case branch that returned only 짹1 based on Math.round(y) for all negative x when |y| >= 2^52, and instead used the normal integer test/parity logic for negative bases. This change is necessary because the pre-patch hardcodes results valid only for x = -1, whereas for general negative x the result magnitude must be |x|^y, not always 1.

**Verdict:** Because the refinement change is NECESSARY and replaces genuinely incorrect, test-specific logic, the pre-patch is a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -100,5 +99,10 @@
 // y is an even integer in this case
 if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
+long yi = Math.round(y);
+if ((yi & 1) == 1) {
+return -1.0;
+} else {
+return 1.0;
+}
 }
 
```

**After OD (correct):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -99,8 +98,4 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
-}
-
 if (y == (long) y) {
 // If y is an integer
@@ -155,5 +150,4 @@
 
 final double result = exp(lna, z, null);
-//result = result + result * z;
 return result;
 }
```

**Ground truth fix:**
```diff
@@ -99,5 +99,5 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+if (y >= TWO_POWER_53 || y <= -TWO_POWER_53) {
 return pow(-x, y);
 }
```

### Patch 3

**Root cause:** The bug is that for negative bases, the code uses 2^52 as the threshold for assuming a floating-point exponent is an exact integer with known parity, but doubles only guarantee all integers are exactly representable up to 2^53, so large odd integers between 2^52 and 2^53 are misclassified as even.

**Why pre-patch overfits:** The pre-patch does address the original failing test by routing the specific exponent through integer-parity handling, but it does so with an additional overbroad heuristic that is not part of the correct fix. For reasonable inputs such as a negative base with a non-integer exponent extremely close to an integer (e.g. `x = -2`, `y = 3.0000000000000004` or other values within the hardcoded tolerance), it can incorrectly return a finite signed result instead of `NaN`.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch's extra condition `Math.abs(y - (long)y) < 1e-15`, which treated near-integer exponents as integers for negative bases. That tolerance-based rule is semantically wrong because `pow(negative, non-integer)` should return `NaN`, and the pre-patch can instead return a signed real value by forcing parity from `(long) y`.

**Verdict:** Because the refinement change is NECESSARY and removes genuinely incorrect tolerance-based logic, the pre-patch is a true overfitting patch. It passes the observed tests but is semantically wrong on other negative-base/non-integer exponent cases.

**Before OD (overfitting):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -99,6 +98,6 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
+if (y >= TWO_POWER_52 || y <= -TWO_POWER_52 || Math.abs(y - (long)y) < 1e-15) {
+return ((long)y & 1) == 0 ? pow(-x, y) : -pow(-x, y);
 }
 
```

**After OD (correct):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -99,8 +98,4 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
-}
-
 if (y == (long) y) {
 // If y is an integer
```

**Ground truth fix:**
```diff
@@ -99,5 +99,5 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+if (y >= TWO_POWER_53 || y <= -TWO_POWER_53) {
 return pow(-x, y);
 }
```

### Patch 4

**Root cause:** The bug is that for negative bases, the code incorrectly assumes any exponent with magnitude at least 2^52 is an even integer, but doubles only guarantee integer/parity exactness beyond 2^53, causing wrong sign handling for values like 5.000000000000001e15.

**Why pre-patch overfits:** The pre-patch changes the buggy threshold logic, but it replaces the original incorrect 2^52 cutoff with another incorrect rule: for |y| > 2^52 it forces parity from a long cast instead of relying on exact integer comparison. That still fails on reasonable inputs such as x = -1 and y = 2^52 + 1, where the exponent is an exactly representable odd integer but the cast-based parity path yields the wrong sign.

**OD refinement (NECESSARY):** The OD refinement removed the special branch that treated |y| > 2^52 as automatically integer and determined parity via ((long) y & 1). This change is necessary because in the interval (2^52, 2^53), doubles can represent odd integers exactly, so the pre-patch still misclassifies some exponents and returns the wrong sign for negative bases.

**Verdict:** Because the refinement change is NECESSARY, the pre-patch remains semantically wrong beyond the triggering test and does not fully fix the root cause. The refined patch removes the unsound >2^52 shortcut, while the pre-patch still overfits by using incorrect parity logic in that range.

**Before OD (overfitting):**
```diff
@@ -99,6 +99,6 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
+if (y > TWO_POWER_52 || y < -TWO_POWER_52) {
+return ((long)y & 1) == 0 ? pow(-x, y) : -pow(-x, y);
 }
 
```

**After OD (correct):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -99,8 +98,4 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
-}
-
 if (y == (long) y) {
 // If y is an integer
@@ -155,5 +150,4 @@
 
 final double result = exp(lna, z, null);
-//result = result + result * z;
 return result;
 }
```

**Ground truth fix:**
```diff
@@ -99,5 +99,5 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+if (y >= TWO_POWER_53 || y <= -TWO_POWER_53) {
 return pow(-x, y);
 }
```

### Patch 5

**Root cause:** The bug is that for negative bases, the code assumes any exponent with magnitude at least 2^52 is an even integer, but doubles only guarantee loss of fractional precision beyond 2^53, so some large odd integer exponents are misclassified and produce the wrong sign.

**Why pre-patch overfits:** The pre-patch does not correctly fix the root cause in a general way: although it makes the failing large-odd-integer test pass, it broadens the negative-base case incorrectly by treating any exponent whose floating-point remainder mod 2 is 1 as odd-integer-like. As a result, inputs such as `pow(-2, 3.5)` would incorrectly return `-pow(2, 3.5)` instead of `NaN`, so it fails on reasonable inputs beyond the provided tests.

**OD refinement (NECESSARY):** The OD refinement removed the pre-patch condition `y >= TWO_POWER_52 || y <= -TWO_POWER_52 || Math.abs(y) % 2 == 1` and replaced it with a single correct integer check `y == (long) y`, then determines parity from the cast long. This is necessary because the pre-patch's extra `Math.abs(y) % 2 == 1` branch forces a negative result for any odd-valued exponent, including small non-integers like 3.5, which should yield NaN for a negative base.

**Verdict:** The refinement is NECESSARY because it restores the required 'exponent must be an integer' guard before applying odd/even sign logic. Since the pre-patch adds semantically wrong behavior for non-integer exponents, it is a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -99,6 +99,6 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
-return pow(-x, y);
+if (y >= TWO_POWER_52 || y <= -TWO_POWER_52 || Math.abs(y) % 2 == 1) {
+return -pow(-x, y);
 }
 
```

**After OD (correct):**
```diff
@@ -9,5 +9,4 @@
 return x;
 }
-
 
 if (x == 0) {
@@ -98,12 +97,14 @@
 /* Handle special case x<0 */
 if (x < 0) {
-// y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+// y is an integer in this case
+if (y == (long) y) {
+long yi = (long) y;
+if ((yi & 1) == 0) {
+// y is even
 return pow(-x, y);
+} else {
+// y is odd
+return -pow(-x, y);
 }
-
-if (y == (long) y) {
-// If y is an integer
-return ((long)y & 1) == 0 ? pow(-x, y) : -pow(-x, y);
 } else {
 return Double.NaN;
@@ -155,5 +156,4 @@
 
 final double result = exp(lna, z, null);
-//result = result + result * z;
 return result;
 }
```

**Ground truth fix:**
```diff
@@ -99,5 +99,5 @@
 if (x < 0) {
 // y is an even integer in this case
-if (y >= TWO_POWER_52 || y <= -TWO_POWER_52) {
+if (y >= TWO_POWER_53 || y <= -TWO_POWER_53) {
 return pow(-x, y);
 }
```

---

## Math-62

**Root cause:** The bug is that multi-start optimization should keep the original [min, max] search interval fixed and vary only the start value, but the buggy code instead randomized the interval bounds and ignored the provided startValue.

**Why pre-patch overfits:** The pre-patch only partially fixes the issue by introducing a start value, but it still randomizes bound1/bound2 for later starts, so each restart may optimize over a different sub-interval rather than the requested full interval. That can miss the true optimum or change behavior on reasonable inputs outside the tests, so it does not correctly implement the root cause fix.

**OD refinement (NECESSARY):** The refinement removed the randomized sub-interval bounds and always called the underlying optimizer with the full [min, max] interval, varying only the start point (and clamping it into range). This is a real semantic correction because the pre-patch still changes the optimization domain on each restart, which is contrary to the intended API and ground-truth fix.

**Verdict:** Because the OD refinement made a necessary semantic change?봱estoring the invariant that all starts use the same [min, max] interval?봳he pre-patch was not already correct. Its test success was due to an incomplete fix that still preserved the original erroneous randomized-bounds behavior.

**Before OD (overfitting):**
```java
// --- L142 ---
public UnivariateRealPointValuePair optimize(final FUNC f, final GoalType goal, final double min, final double max) throws FunctionEvaluationException { return optimize(f, goal, min, max, min + generator.nextDouble() * (max - min)); }

// --- L150 ---
public UnivariateRealPointValuePair optimize(final FUNC f, final GoalType goal, final double min, final double max, final double startValue) throws FunctionEvaluationException { optima = new UnivariateRealPointValuePair[starts]; totalEvaluations = 0; for (int i = 0; i < starts; ++i) { try { final double bound1 = (i == 0) ? min : min + generator.nextDouble() * (max - min); final double bound2 = (i == 0) ? max : min + generator.nextDouble() * (max - min); final double actualStart = (i == 0) ? startValue : min + generator.nextDouble() * (max - min); optima[i] = optimizer.optimize(f, goal, FastMath.min(bound1, bound2), FastMath.max(bound1, bound2), actualStart); } catch (FunctionEvaluationException fee) { optima[i] = null; } catch (ConvergenceException ce) { optima[i] = null; } final int usedEvaluations = optimizer.getEvaluations(); optimizer.setMaxEvaluations(optimizer.getMaxEvaluations() - usedEvaluations); totalEvaluations += usedEvaluations; } sortPairs(goal); if (optima[0] == null) { throw new ConvergenceException(LocalizedFormats.NO_CONVERGENCE_WITH_ANY_START_POINT, starts); } return optima[0]; }
```

**After OD (correct):**
```java
// --- L142 ---
public UnivariateRealPointValuePair optimize(final FUNC f, final GoalType goal, final double min, final double max) throws FunctionEvaluationException { return optimize(f, goal, min, max, min + generator.nextDouble() * (max - min)); }

// --- L150 ---
public UnivariateRealPointValuePair optimize(final FUNC f, final GoalType goal, final double min, final double max, final double startValue) throws FunctionEvaluationException { optima = new UnivariateRealPointValuePair[starts]; totalEvaluations = 0; for (int i = 0; i < starts; ++i) { try { final double actualStart = Math.min(max, Math.max(min, (i == 0) ? startValue : min + generator.nextDouble() * (max - min))); optima[i] = optimizer.optimize(f, goal, min, max, actualStart); } catch (FunctionEvaluationException fee) { optima[i] = null; } catch (ConvergenceException ce) { optima[i] = null; } final int usedEvaluations = optimizer.getEvaluations(); optimizer.setMaxEvaluations(optimizer.getMaxEvaluations() - usedEvaluations); totalEvaluations += usedEvaluations; } sortPairs(goal); if (optima[0] == null) { throw new ConvergenceException(LocalizedFormats.NO_CONVERGENCE_WITH_ANY_START_POINT, starts); } return optima[0]; }
```

---

## Math-94

**Root cause:** The bug is the zero-input guard `if (u * v == 0)`, whose multiplication can overflow to 0 for nonzero inputs (e.g., large powers of two), causing the method to return `|u|+|v|` instead of computing the gcd.

**Why pre-patch overfits:** The pre-patch does fix the actual reported root cause by replacing the overflow-prone `u * v == 0` test with explicit zero checks, so it correctly handles the failing test and the intended zero cases. However, it also removes required overflow protection, so it is not fully correct on reasonable edge inputs such as `gcd(Integer.MIN_VALUE, 0)`-style nonzero paths avoided by the early return or especially `gcd(Integer.MIN_VALUE, Integer.MIN_VALUE)`, where it should throw but instead returns an invalid negative value.

**OD refinement (NECESSARY):** The OD refinement restored the `k < 31` loop bound and the subsequent `if (k == 31) throw new ArithmeticException(...)` overflow check in the binary-gcd reduction step. This is semantically necessary because without it, inputs whose gcd is 2^31 (not representable as a positive int) are mishandled: the pre-patch can shift 31 times and then compute `1 << 31`, yielding an incorrect negative result instead of the specified overflow exception.

**Verdict:** Because the refinement change is NECESSARY and adds missing overflow-handling logic that the pre-patch removed, the pre-patch is semantically incomplete. It passes the available tests but fails on valid edge cases outside them, so it is a true overfitting patch.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,9 @@
 public static int gcd(int u, int v) {
-if (u * v == 0) {
-return (Math.abs(u) + Math.abs(v));
+if (u == 0 && v == 0) {
+return 0;
+} else if (u == 0) {
+return Math.abs(v);
+} else if (v == 0) {
+return Math.abs(u);
 }
 // keep u and v negative, as negative integers range down to
@@ -16,12 +20,9 @@
 // B1. [Find power of 2]
 int k = 0;
-while ((u & 1) == 0 && (v & 1) == 0 && k < 31) { // while u and v are
+while ((u & 1) == 0 && (v & 1) == 0) { // while u and v are
 // both even...
 u /= 2;
 v /= 2;
 k++; // cast out twos.
-}
-if (k == 31) {
-throw new ArithmeticException("overflow: gcd is 2^31");
 }
 // B2. Initialize: u and v have been divided by 2^k and at least
```

**After OD (correct):**
```diff
@@ -1,5 +1,9 @@
 public static int gcd(int u, int v) {
-if (u * v == 0) {
-return (Math.abs(u) + Math.abs(v));
+if (u == 0 && v == 0) {
+return 0;
+} else if (u == 0) {
+return Math.abs(v);
+} else if (v == 0) {
+return Math.abs(u);
 }
 // keep u and v negative, as negative integers range down to
```

**Ground truth fix:**
```diff
@@ -1,4 +1,4 @@
 public static int gcd(int u, int v) {
-if (u * v == 0) {
+if ((u == 0) || (v == 0)) {
 return (Math.abs(u) + Math.abs(v));
 }
```

---

## Mockito-22

### Patch 1

**Root cause:** The bug is that areEqual calls o1.equals(o2) even when o1 and o2 are the exact same object, so a broken equals implementation can throw instead of returning true for identical references.

**Why pre-patch overfits:** The pre-patch partially addresses the failing BadEquals case by short-circuiting on identity in the final branch, but it changes semantics by requiring o1 and o2 to have the same runtime class before equals is consulted. That breaks reasonable inputs beyond the tests, such as two non-array objects of different classes whose equals method legitimately returns true, so it does not correctly preserve the intended equality behavior.

**OD refinement (NECESSARY):** The refinement adds an unconditional identity check at the top (if (o1 == o2) return true) and removes the restrictive same-class guard before calling equals. This is necessary because the pre-patch only treats identical references as equal in the non-array branch and incorrectly rejects valid cross-class equality cases that the original method and ground-truth fix are supposed to support.

**Verdict:** Because the refinement change is NECESSARY, not cosmetic, and the pre-patch introduced an extra same-class restriction absent from the correct fix, the pre-patch is an overfitting/semantically wrong patch rather than a genuinely correct alternative.

**Before OD (overfitting):**
```diff
@@ -5,5 +5,5 @@
 return isArray(o2) && areArraysEqual(o1, o2);
 } else {
-return o1.equals(o2);
+return o1 == o2 || (o1 != null && o1.getClass().equals(o2.getClass()) && o1.equals(o2));
 }
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
-return o1 == null && o2 == null;
+if (o1 == o2) {
+return true;
+} else if (o1 == null || o2 == null) {
+return false;
 } else if (isArray(o1)) {
 return isArray(o2) && areArraysEqual(o1, o2);
```

**Ground truth fix:**
```diff
@@ -1,4 +1,6 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
+if (o1 == o2 ) {
+return true;
+lse if (o1 == null || o2 == null) {
 return o1 == null && o2 == null;
 } else if (isArray(o1)) {
```

### Patch 2

**Root cause:** The bug is that areEqual calls o1.equals(o2) even when o1 and o2 are the exact same object, so self-comparison can throw from a broken equals implementation instead of returning true by identity.

**Why pre-patch overfits:** The pre-patch does address the immediate failing case by checking o1 == o2 before calling equals, but it also introduces a new semantic restriction: non-array objects are only compared with equals when their classes are exactly equal. That is not part of the original contract and would fail on reasonable inputs such as two different classes with a valid symmetric cross-class equals implementation, so it is an overfitting patch.

**OD refinement (NECESSARY):** The refinement removed the pre-patch's extra same-class restriction and restored the original general equality semantics after adding the needed identity check first. This change is necessary because the pre-patch incorrectly returns false for non-array objects of different runtime classes even when their equals method would legitimately return true, so it changes behavior beyond the bug fix.

**Verdict:** Because the refinement change is NECESSARY, not cosmetic, the pre-patch was missing correct general equality behavior and added an unjustified class-equality guard. Therefore the original patch passed tests but was semantically wrong on broader valid inputs.

**Before OD (overfitting):**
```diff
@@ -5,5 +5,11 @@
 return isArray(o2) && areArraysEqual(o1, o2);
 } else {
+if (o1 == o2) {
+return true;
+} else if (o1 != null && o2 != null && o1.getClass().equals(o2.getClass())) {
 return o1.equals(o2);
+} else {
+return false;
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
-return o1 == null && o2 == null;
+if (o1 == o2) {
+return true;
+} else if (o1 == null || o2 == null) {
+return false;
 } else if (isArray(o1)) {
 return isArray(o2) && areArraysEqual(o1, o2);
```

**Ground truth fix:**
```diff
@@ -1,4 +1,6 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
+if (o1 == o2 ) {
+return true;
+lse if (o1 == null || o2 == null) {
 return o1 == null && o2 == null;
 } else if (isArray(o1)) {
```

### Patch 3

**Root cause:** The bug is that areEqual calls o1.equals(o2) even when o1 and o2 are the exact same object, so a broken equals implementation can throw instead of returning true for identical references.

**Why pre-patch overfits:** The pre-patch does not fully fix the root cause; it merely catches exceptions from equals and falls back to identity, which is narrower and changes behavior for any Throwable not subclassing Exception. It also changes array semantics by requiring both operands to be arrays before entering array comparison, whereas the intended logic is driven by whether the left operand is an array; although many cases still return the same result, the patch is not the proper general fix.

**OD refinement (NECESSARY):** The refinement adds an explicit identity check (o1 == o2) before any null, array, or equals logic, and restores the original asymmetric array handling via isArray(o1). This is necessary because the pre-patch only recovers from exceptions thrown by equals, but does not implement the correct semantic rule that identical references must always compare equal without invoking equals.

**Verdict:** Because the OD refinement introduced necessary logic?봞n unconditional fast path for identical references that avoids calling equals at all?봳he pre-patch was not semantically correct and was overfitting to the observed failing test via exception catching.

**Before OD (overfitting):**
```diff
@@ -2,8 +2,12 @@
 if (o1 == null || o2 == null) {
 return o1 == null && o2 == null;
-} else if (isArray(o1)) {
-return isArray(o2) && areArraysEqual(o1, o2);
+} else if (o1.getClass().isArray() && o2.getClass().isArray()) {
+return areArraysEqual(o1, o2);
 } else {
+try {
 return o1.equals(o2);
+} catch (Exception e) {
+return o1 == o2;
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
-return o1 == null && o2 == null;
+if (o1 == o2) {
+return true;
+} else if (o1 == null || o2 == null) {
+return false;
 } else if (isArray(o1)) {
 return isArray(o2) && areArraysEqual(o1, o2);
```

**Ground truth fix:**
```diff
@@ -1,4 +1,6 @@
 public static boolean areEqual(Object o1, Object o2) {
-if (o1 == null || o2 == null) {
+if (o1 == o2 ) {
+return true;
+lse if (o1 == null || o2 == null) {
 return o1 == null && o2 == null;
 } else if (isArray(o1)) {
```

---

## Mockito-5

### Patch 1

**Root cause:** The bug is that VerificationOverTimeImpl explicitly catches a JUnit-specific assertion subtype, creating an unwanted runtime dependency on JUnit instead of handling assertion failures generically.

**Why pre-patch overfits:** The pre-patch removes the JUnit dependency, so it fixes the specific failing test, but it does not fully fix the method's intended behavior. On reasonable inputs where delegate.verify throws an AssertionError that is neither MockitoAssertionError nor the removed JUnit-specific class, the pre-patch would fail to retry/record the error correctly and would throw prematurely.

**OD refinement (NECESSARY):** The refinement broadened the catch from only MockitoAssertionError to all AssertionError, restoring the original behavior of retrying/handling non-Mockito assertion failures while still avoiding any JUnit-specific type reference. This is necessary because simply deleting the JUnit-specific catch changes semantics: other AssertionError subtypes would now escape immediately instead of being processed by handleVerifyException.

**Verdict:** Because the refinement introduced necessary missing logic?봥eneric handling of AssertionError?봳he pre-patch was not semantically equivalent to the correct fix. It passed tests by removing the dependency but overfit by dropping required assertion-handling behavior.

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

**Root cause:** The bug is that VerificationOverTimeImpl directly references a JUnit-specific assertion exception type, creating an unwanted runtime dependency on JUnit instead of handling assertion failures generically.

**Why pre-patch overfits:** The pre-patch fixes the immediate class-loading problem by removing the JUnit-specific reference, so the failing dependency test passes. However, it changes behavior: if delegate.verify(data) throws an AssertionError that is not a MockitoAssertionError, it will now escape immediately instead of being captured and retried over time, which is inconsistent with the intended generic assertion-handling behavior of the ground-truth fix.

**OD refinement (NECESSARY):** The refinement replaced the narrow catch of MockitoAssertionError with a catch of AssertionError, thereby preserving retry/timeout handling for all assertion failures while removing the JUnit-specific type reference. This is necessary because the pre-patch only deletes the JUnit-specific catch, which avoids the dependency but stops handling other AssertionError subclasses that should still be retried and processed by handleVerifyException.

**Verdict:** Because the OD refinement introduced necessary missing logic?봥eneric AssertionError handling?봳he pre-patch was not semantically complete. It passed tests by removing the offending dependency but overfit by dropping required verification behavior for reasonable non-JUnit assertion failures.

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

---

