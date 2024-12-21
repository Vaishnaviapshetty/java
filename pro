//two sum
class Solution {
  public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> numToIndex = new HashMap<>();

    for (int i = 0; i < nums.length; ++i) {
      if (numToIndex.containsKey(target - nums[i]))
        return new int[] {numToIndex.get(target - nums[i]), i};
      numToIndex.put(nums[i], i);
    }

    throw new IllegalArgumentException();
  }
}

//add two numbers
class Solution {
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    int carry = 0;

    while (l1 != null || l2 != null || carry > 0) {
      if (l1 != null) {
        carry += l1.val;
        l1 = l1.next;
      }
      if (l2 != null) {
        carry += l2.val;
        l2 = l2.next;
      }
      curr.next = new ListNode(carry % 10);
      carry /= 10;
      curr = curr.next;
    }

    return dummy.next;
  }
}

//longest sybstring without repeating characters
class Solution {
  public int lengthOfLongestSubstring(String s) {
    int ans = 0;
    int[] count = new int[128];

    for (int l = 0, r = 0; r < s.length(); ++r) {
      ++count[s.charAt(r)];
      while (count[s.charAt(r)] > 1)
        --count[s.charAt(l++)];
      ans = Math.max(ans, r - l + 1);
    }

    return ans;
  }
}

//medain of two sorted arrays
class Solution {
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    final int n1 = nums1.length;
    final int n2 = nums2.length;
    if (n1 > n2)
      return findMedianSortedArrays(nums2, nums1);

    int l = 0;
    int r = n1;

    while (l <= r) {
      final int partition1 = (l + r) / 2;
      final int partition2 = (n1 + n2 + 1) / 2 - partition1;
      final int maxLeft1 = partition1 == 0 ? Integer.MIN_VALUE : nums1[partition1 - 1];
      final int maxLeft2 = partition2 == 0 ? Integer.MIN_VALUE : nums2[partition2 - 1];
      final int minRight1 = partition1 == n1 ? Integer.MAX_VALUE : nums1[partition1];
      final int minRight2 = partition2 == n2 ? Integer.MAX_VALUE : nums2[partition2];
      if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1)
        return (n1 + n2) % 2 == 0
            ? (Math.max(maxLeft1, maxLeft2) + Math.min(minRight1, minRight2)) * 0.5
            : Math.max(maxLeft1, maxLeft2);
      else if (maxLeft1 > minRight2)
        r = partition1 - 1;
      else
        l = partition1 + 1;
    }

    throw new IllegalArgumentException();
  }
}

//longest palindromic substring
class Solution {
  public String longestPalindrome(String s) {
    if (s.isEmpty())
      return "";

    // (start, end) indices of the longest palindrome in s
    int[] indices = {0, 0};

    for (int i = 0; i < s.length(); ++i) {
      int[] indices1 = extend(s, i, i);
      if (indices1[1] - indices1[0] > indices[1] - indices[0])
        indices = indices1;
      if (i + 1 < s.length() && s.charAt(i) == s.charAt(i + 1)) {
        int[] indices2 = extend(s, i, i + 1);
        if (indices2[1] - indices2[0] > indices[1] - indices[0])
          indices = indices2;
      }
    }

    return s.substring(indices[0], indices[1] + 1);
  }

  // Returns the (start, end) indices of the longest palindrome extended from
  // the substring s[i..j].
  private int[] extend(final String s, int i, int j) {
    for (; i >= 0 && j < s.length(); --i, ++j)
      if (s.charAt(i) != s.charAt(j))
        break;
    return new int[] {i + 1, j - 1};
  }
}

//zigzag conversion
class Solution {
  public String convert(String s, int numRows) {
    StringBuilder sb = new StringBuilder();
    List<Character>[] rows = new List[numRows];
    int k = 0;
    int direction = numRows == 1 ? 0 : -1;

    for (int i = 0; i < numRows; ++i)
      rows[i] = new ArrayList<>();

    for (final char c : s.toCharArray()) {
      rows[k].add(c);
      if (k == 0 || k == numRows - 1)
        direction *= -1;
      k += direction;
    }

    for (List<Character> row : rows)
      for (final char c : row)
        sb.append(c);

    return sb.toString();
  }
}

//reverse integer
class Solution {
  public int reverse(int x) {
    long ans = 0;

    while (x != 0) {
      ans = ans * 10 + x % 10;
      x /= 10;
    }

    return (ans < Integer.MIN_VALUE || ans > Integer.MAX_VALUE) ? 0 : (int) ans;
  }
}

//string to integer(atoi)
class Solution {
  public int myAtoi(String s) {
    s = s.strip();
    if (s.isEmpty())
      return 0;

    final int sign = s.charAt(0) == '-' ? -1 : 1;
    if (s.charAt(0) == '+' || s.charAt(0) == '-')
      s = s.substring(1);

    long num = 0;

    for (final char c : s.toCharArray()) {
      if (!Character.isDigit(c))
        break;
      num = num * 10 + (c - '0');
      if (sign * num <= Integer.MIN_VALUE)
        return Integer.MIN_VALUE;
      if (sign * num >= Integer.MAX_VALUE)
        return Integer.MAX_VALUE;
    }

    return sign * (int) num;
  }
}

//palindrome number
class Solution {
  public boolean isPalindrome(int x) {
    if (x < 0)
      return false;

    long reversed = 0;
    int y = x;

    while (y > 0) {
      reversed = reversed * 10 + y % 10;
      y /= 10;
    }

    return reversed == x;
  }
}

//regular expression matching
class Solution {
  public boolean isMatch(String s, String p) {
    final int m = s.length();
    final int n = p.length();
    // dp[i][j] := true if s[0..i) matches p[0..j)
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;

    for (int j = 0; j < p.length(); ++j)
      if (p.charAt(j) == '*' && dp[0][j - 1])
        dp[0][j + 1] = true;

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        if (p.charAt(j) == '*') {
          // The minimum index of '*' is 1.
          final boolean noRepeat = dp[i + 1][j - 1];
          final boolean doRepeat = isMatch(s, i, p, j - 1) && dp[i][j + 1];
          dp[i + 1][j + 1] = noRepeat || doRepeat;
        } else if (isMatch(s, i, p, j)) {
          dp[i + 1][j + 1] = dp[i][j];
        }

    return dp[m][n];
  }

  private boolean isMatch(final String s, int i, final String p, int j) {
    return j >= 0 && p.charAt(j) == '.' || s.charAt(i) == p.charAt(j);
  }
}

//container with most water
class Solution {
  public int maxArea(int[] height) {
    int ans = 0;
    int l = 0;
    int r = height.length - 1;

    while (l < r) {
      final int minHeight = Math.min(height[l], height[r]);
      ans = Math.max(ans, minHeight * (r - l));
      if (height[l] < height[r])
        ++l;
      else
        --r;
    }

    return ans;
  }
}

//integer to roman
class Solution {
  public String intToRoman(int num) {
    final int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    final String[] symbols = {"M",  "CM", "D",  "CD", "C",  "XC", "L",
                              "XL", "X",  "IX", "V",  "IV", "I"};
    StringBuilder sb = new StringBuilder();

    for (int i = 0; i < values.length; ++i) {
      if (num == 0)
        break;
      while (num >= values[i]) {
        num -= values[i];
        sb.append(symbols[i]);
      }
    }

    return sb.toString();
  }
}

//roman to integer
class Solution {
  public int romanToInt(String s) {
    int ans = 0;
    int[] roman = new int[128];

    roman['I'] = 1;
    roman['V'] = 5;
    roman['X'] = 10;
    roman['L'] = 50;
    roman['C'] = 100;
    roman['D'] = 500;
    roman['M'] = 1000;

    for (int i = 0; i + 1 < s.length(); ++i)
      if (roman[s.charAt(i)] < roman[s.charAt(i + 1)])
        ans -= roman[s.charAt(i)];
      else
        ans += roman[s.charAt(i)];

    return ans + roman[s.charAt(s.length() - 1)];
  }
}

//longestcommom prefix
class Solution {
  public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0)
      return "";

    for (int i = 0; i < strs[0].length(); ++i)
      for (int j = 1; j < strs.length; ++j)
        if (i == strs[j].length() || strs[j].charAt(i) != strs[0].charAt(i))
          return strs[0].substring(0, i);

    return strs[0];
  }
}
//3sum
class Solution {
  public List<List<Integer>> threeSum(int[] nums) {
    if (nums.length < 3)
      return new ArrayList<>();

    List<List<Integer>> ans = new ArrayList<>();

    Arrays.sort(nums);

    for (int i = 0; i + 2 < nums.length; ++i) {
      if (i > 0 && nums[i] == nums[i - 1])
        continue;
      // Choose nums[i] as the first number in the triplet, then search the
      // remaining numbers in [i + 1, n - 1].
      int l = i + 1;
      int r = nums.length - 1;
      while (l < r) {
        final int sum = nums[i] + nums[l] + nums[r];
        if (sum == 0) {
          ans.add(Arrays.asList(nums[i], nums[l++], nums[r--]));
          while (l < r && nums[l] == nums[l - 1])
            ++l;
          while (l < r && nums[r] == nums[r + 1])
            --r;
        } else if (sum < 0) {
          ++l;
        } else {
          --r;
        }
      }
    }

    return ans;
  }
}

//3sum closet
class Solution {
  public int threeSumClosest(int[] nums, int target) {
    int ans = nums[0] + nums[1] + nums[2];

    Arrays.sort(nums);

    for (int i = 0; i + 2 < nums.length; ++i) {
      if (i > 0 && nums[i] == nums[i - 1])
        continue;
      // Choose nums[i] as the first number in the triplet, then search the
      // remaining numbers in [i + 1, n - 1].
      int l = i + 1;
      int r = nums.length - 1;
      while (l < r) {
        final int sum = nums[i] + nums[l] + nums[r];
        if (sum == target)
          return sum;
        if (Math.abs(sum - target) < Math.abs(ans - target))
          ans = sum;
        if (sum < target)
          ++l;
        else
          --r;
      }
    }

    return ans;
  }
}

//letter comination of a phone number
class Solution {
  public List<String> letterCombinations(String digits) {
    if (digits.isEmpty())
      return new ArrayList<>();

    List<String> ans = new ArrayList<>();

    dfs(digits, 0, new StringBuilder(), ans);
    return ans;
  }

  private static final String[] digitToLetters = {"",    "",    "abc",  "def", "ghi",
                                                  "jkl", "mno", "pqrs", "tuv", "wxyz"};

  private void dfs(String digits, int i, StringBuilder sb, List<String> ans) {
    if (i == digits.length()) {
      ans.add(sb.toString());
      return;
    }

    for (final char c : digitToLetters[digits.charAt(i) - '0'].toCharArray()) {
      sb.append(c);
      dfs(digits, i + 1, sb, ans);
      sb.deleteCharAt(sb.length() - 1);
    }
  }
}

//4sum
class Solution {
  public List<List<Integer>> fourSum(int[] nums, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(nums);
    nSum(nums, 4, target, 0, nums.length - 1, new ArrayList<>(), ans);
    return ans;
  }

  // Finds n numbers that add up to the target in [l, r].
  private void nSum(int[] nums, long n, long target, int l, int r, List<Integer> path,
                    List<List<Integer>> ans) {
    if (r - l + 1 < n || target < nums[l] * n || target > nums[r] * n)
      return;
    if (n == 2) {
      // Similar to the sub procedure in 15. 3Sum
      while (l < r) {
        final int sum = nums[l] + nums[r];
        if (sum == target) {
          path.add(nums[l]);
          path.add(nums[r]);
          ans.add(new ArrayList<>(path));
          path.remove(path.size() - 1);
          path.remove(path.size() - 1);
          ++l;
          --r;
          while (l < r && nums[l] == nums[l - 1])
            ++l;
          while (l < r && nums[r] == nums[r + 1])
            --r;
        } else if (sum < target) {
          ++l;
        } else {
          --r;
        }
      }
      return;
    }

    for (int i = l; i <= r; ++i) {
      if (i > l && nums[i] == nums[i - 1])
        continue;
      path.add(nums[i]);
      nSum(nums, n - 1, target - nums[i], i + 1, r, path, ans);
      path.remove(path.size() - 1);
    }
  }
}

//removeNth node from end of the list
class Solution {
  public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode slow = head;
    ListNode fast = head;

    while (n-- > 0)
      fast = fast.next;
    if (fast == null)
      return head.next;

    while (fast.next != null) {
      slow = slow.next;
      fast = fast.next;
    }
    slow.next = slow.next.next;

    return head;
  }
}

//valid parentheses
class Solution {
  public boolean isValid(String s) {
    Deque<Character> stack = new ArrayDeque<>();

    for (final char c : s.toCharArray())
      if (c == '(')
        stack.push(')');
      else if (c == '{')
        stack.push('}');
      else if (c == '[')
        stack.push(']');
      else if (stack.isEmpty() || stack.pop() != c)
        return false;

    return stack.isEmpty();
  }
}

//merge two sorted lists
class Solution {
  public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    if (list1 == null || list2 == null)
      return list1 == null ? list2 : list1;
    if (list1.val > list2.val) {
      ListNode temp = list1;
      list1 = list2;
      list2 = temp;
    }
    list1.next = mergeTwoLists(list1.next, list2);
    return list1;
  }
}

//generate parentheses
class Solution {
  public List<String> generateParenthesis(int n) {
    List<String> ans = new ArrayList<>();
    dfs(n, n, new StringBuilder(), ans);
    return ans;
  }

  private void dfs(int l, int r, StringBuilder sb, List<String> ans) {
    if (l == 0 && r == 0) {
      ans.add(sb.toString());
      return;
    }

    if (l > 0) {
      sb.append("(");
      dfs(l - 1, r, sb, ans);
      sb.deleteCharAt(sb.length() - 1);
    }
    if (l < r) {
      sb.append(")");
      dfs(l, r - 1, sb, ans);
      sb.deleteCharAt(sb.length() - 1);
    }
  }
}

//merge k sorted lists
class Solution {
  public ListNode mergeKLists(ListNode[] lists) {
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    Queue<ListNode> minHeap = new PriorityQueue<>((a, b) -> Integer.compare(a.val, b.val));

    for (final ListNode list : lists)
      if (list != null)
        minHeap.offer(list);

    while (!minHeap.isEmpty()) {
      ListNode minNode = minHeap.poll();
      if (minNode.next != null)
        minHeap.offer(minNode.next);
      curr.next = minNode;
      curr = curr.next;
    }

    return dummy.next;
  }
}

//swap nodes in paris
class Solution {
  public ListNode swapPairs(ListNode head) {
    final int length = getLength(head);
    ListNode dummy = new ListNode(0, head);
    ListNode prev = dummy;
    ListNode curr = head;

    for (int i = 0; i < length / 2; ++i) {
      ListNode next = curr.next;
      curr.next = next.next;
      next.next = curr;
      prev.next = next;
      prev = curr;
      curr = curr.next;
    }

    return dummy.next;
  }

  private int getLength(ListNode head) {
    int length = 0;
    for (ListNode curr = head; curr != null; curr = curr.next)
      ++length;
    return length;
  }
}

//reverse nodes in k-group
class Solution {
  public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null)
      return null;

    ListNode tail = head;

    for (int i = 0; i < k; ++i) {
      // There are less than k nodes in the list, do nothing.
      if (tail == null)
        return head;
      tail = tail.next;
    }

    ListNode newHead = reverse(head, tail);
    head.next = reverseKGroup(tail, k);
    return newHead;
  }

  // Reverses [head, tail).
  private ListNode reverse(ListNode head, ListNode tail) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != tail) {
      ListNode next = curr.next;
      curr.next = prev;
      prev = curr;
      curr = next;
    }
    return prev;
  }
}

//remove duplicates from sorted array
class Solution {
  public int removeDuplicates(int[] nums) {
    int i = 0;

    for (final int num : nums)
      if (i < 1 || num > nums[i - 1])
        nums[i++] = num;

    return i;
  }
}

//remove element
class Solution {
  public int removeElement(int[] nums, int val) {
    int i = 0;

    for (final int num : nums)
      if (num != val)
        nums[i++] = num;

    return i;
  }
}

//find the index of the first occurrence in a string
class Solution {
  public int strStr(String haystack, String needle) {
    final int m = haystack.length();
    final int n = needle.length();

    for (int i = 0; i < m - n + 1; ++i)
      if (haystack.substring(i, i + n).equals(needle))
        return i;

    return -1;
  }
}

//divide two integers
class Solution {
  public int divide(long dividend, long divisor) {
    // -2^{31} / -1 = 2^31 will overflow, so return 2^31 - 1.
    if (dividend == Integer.MIN_VALUE && divisor == -1)
      return Integer.MAX_VALUE;

    final int sign = dividend > 0 ^ divisor > 0 ? -1 : 1;
    long ans = 0;
    long dvd = Math.abs(dividend);
    long dvs = Math.abs(divisor);

    while (dvd >= dvs) {
      long k = 1;
      while (k * 2 * dvs <= dvd)
        k *= 2;
      dvd -= k * dvs;
      ans += k;
    }

    return sign * (int) ans;
  }
}

//Substring with Concatenation of All Words
class Solution {
  public List<Integer> findSubstring(String s, String[] words) {
    if (s.isEmpty() || words.length == 0)
      return new ArrayList<>();

    final int k = words.length;
    final int n = words[0].length();
    List<Integer> ans = new ArrayList<>();
    Map<String, Integer> count = new HashMap<>();

    for (final String word : words)
      count.merge(word, 1, Integer::sum);

    for (int i = 0; i <= s.length() - k * n; ++i) {
      Map<String, Integer> seen = new HashMap<>();
      int j = 0;
      for (; j < k; ++j) {
        final String word = s.substring(i + j * n, i + j * n + n);
        seen.merge(word, 1, Integer::sum);
        if (seen.get(word) > count.getOrDefault(word, 0))
          break;
      }
      if (j == k)
        ans.add(i);
    }

    return ans;
  }
}

//next permutation
class Solution {
  public void nextPermutation(int[] nums) {
    final int n = nums.length;

    // From back to front, find the first number < nums[i + 1].
    int i;
    for (i = n - 2; i >= 0; --i)
      if (nums[i] < nums[i + 1])
        break;

    // From back to front, find the first number > nums[i], swap it with
    // nums[i].
    if (i >= 0)
      for (int j = n - 1; j > i; --j)
        if (nums[j] > nums[i]) {
          swap(nums, i, j);
          break;
        }

    // Reverse nums[i + 1..n - 1].
    reverse(nums, i + 1, n - 1);
  }

  private void reverse(int[] nums, int l, int r) {
    while (l < r)
      swap(nums, l++, r--);
  }

  private void swap(int[] nums, int i, int j) {
    final int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }
}

//longest valid parentheses
class Solution {
  public int longestValidParentheses(String s) {
    final String s2 = ")" + s;
    // dp[i] := the length of the longest valid parentheses in the substring
    // s2[1..i]
    int dp[] = new int[s2.length()];

    for (int i = 1; i < s2.length(); ++i)
      if (s2.charAt(i) == ')' && s2.charAt(i - dp[i - 1] - 1) == '(')
        dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2;

    return Arrays.stream(dp).max().getAsInt();
  }
}

//search in rotated sorted array
class Solution {
  public int search(int[] nums, int target) {
    int l = 0;
    int r = nums.length - 1;

    while (l <= r) {
      final int m = (l + r) / 2;
      if (nums[m] == target)
        return m;
      if (nums[l] <= nums[m]) { // nums[l..m] are sorted.
        if (nums[l] <= target && target < nums[m])
          r = m - 1;
        else
          l = m + 1;
      } else { // nums[m..n - 1] are sorted.
        if (nums[m] < target && target <= nums[r])
          l = m + 1;
        else
          r = m - 1;
      }
    }

    return -1;
  }
}

//find first and last position of element in sorted array
class Solution {
  public int[] searchRange(int[] nums, int target) {
    final int l = firstGreaterEqual(nums, target);
    if (l == nums.length || nums[l] != target)
      return new int[] {-1, -1};
    final int r = firstGreaterEqual(nums, target + 1) - 1;
    return new int[] {l, r};
  }

  private int firstGreaterEqual(int[] A, int target) {
    int l = 0;
    int r = A.length;
    while (l < r) {
      final int m = (l + r) / 2;
      if (A[m] >= target)
        r = m;
      else
        l = m + 1;
    }
    return l;
  }
}

//search insert position
class Solution {
  public int searchInsert(int[] nums, int target) {
    int l = 0;
    int r = nums.length;

    while (l < r) {
      final int m = (l + r) / 2;
      if (nums[m] == target)
        return m;
      if (nums[m] < target)
        l = m + 1;
      else
        r = m;
    }

    return l;
  }
}

//valid sudoku 
class Solution {
  public boolean isValidSudoku(char[][] board) {
    Set<String> seen = new HashSet<>();

    for (int i = 0; i < 9; ++i)
      for (int j = 0; j < 9; ++j) {
        if (board[i][j] == '.')
          continue;
        final char c = board[i][j];
        if (!seen.add(c + "@row" + i) || //
            !seen.add(c + "@col" + j) || //
            !seen.add(c + "@box" + i / 3 + j / 3))
          return false;
      }

    return true;
  }
}

//sudoku solver
class Solution {
  public void solveSudoku(char[][] board) {
    dfs(board, 0);
  }

  private boolean dfs(char[][] board, int s) {
    if (s == 81)
      return true;

    final int i = s / 9;
    final int j = s % 9;

    if (board[i][j] != '.')
      return dfs(board, s + 1);

    for (char c = '1'; c <= '9'; ++c)
      if (isValid(board, i, j, c)) {
        board[i][j] = c;
        if (dfs(board, s + 1))
          return true;
        board[i][j] = '.';
      }

    return false;
  }

  private boolean isValid(char[][] board, int row, int col, char c) {
    for (int i = 0; i < 9; ++i)
      if (board[i][col] == c || board[row][i] == c ||
          board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c)
        return false;
    return true;
  }
}

//count and say
class Solution {
  public String countAndSay(int n) {
    StringBuilder sb = new StringBuilder("1");

    while (--n > 0) {
      StringBuilder next = new StringBuilder();
      for (int i = 0; i < sb.length(); ++i) {
        int count = 1;
        while (i + 1 < sb.length() && sb.charAt(i) == sb.charAt(i + 1)) {
          ++count;
          ++i;
        }
        next.append(count).append(sb.charAt(i));
      }
      sb = next;
    }

    return sb.toString();
  }
}

//combination sum
class Solution {
  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(candidates);
    dfs(0, candidates, target, new ArrayList<>(), ans);
    return ans;
  }

  private void dfs(int s, int[] candidates, int target, List<Integer> path,
                   List<List<Integer>> ans) {
    if (target < 0)
      return;
    if (target == 0) {
      ans.add(new ArrayList<>(path));
      return;
    }

    for (int i = s; i < candidates.length; ++i) {
      path.add(candidates[i]);
      dfs(i, candidates, target - candidates[i], path, ans);
      path.remove(path.size() - 1);
    }
  }
}

//combination sum 2
class Solution {
  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(candidates);
    dfs(0, candidates, target, new ArrayList<>(), ans);
    return ans;
  }

  private void dfs(int s, int[] candidates, int target, List<Integer> path,
                   List<List<Integer>> ans) {
    if (target < 0)
      return;
    if (target == 0) {
      ans.add(new ArrayList<>(path));
      return;
    }

    for (int i = s; i < candidates.length; ++i) {
      if (i > s && candidates[i] == candidates[i - 1])
        continue;
      path.add(candidates[i]);
      dfs(i + 1, candidates, target - candidates[i], path, ans);
      path.remove(path.size() - 1);
    }
  }
}

//first missing positive 
class Solution {
  public int firstMissingPositive(int[] nums) {
    final int n = nums.length;

    // Correct slot:
    // nums[i] = i + 1
    // nums[i] - 1 = i
    // nums[nums[i] - 1] = nums[i]
    for (int i = 0; i < n; ++i)
      while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1])
        swap(nums, i, nums[i] - 1);

    for (int i = 0; i < n; ++i)
      if (nums[i] != i + 1)
        return i + 1;

    return n + 1;
  }

  private void swap(int[] nums, int i, int j) {
    final int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }
}

//trapping rain water
class Solution {
  public int trap(int[] height) {
    final int n = height.length;
    int ans = 0;
    int[] l = new int[n]; // l[i] := max(height[0..i])
    int[] r = new int[n]; // r[i] := max(height[i..n))

    for (int i = 0; i < n; ++i)
      l[i] = i == 0 ? height[i] : Math.max(height[i], l[i - 1]);

    for (int i = n - 1; i >= 0; --i)
      r[i] = i == n - 1 ? height[i] : Math.max(height[i], r[i + 1]);

    for (int i = 0; i < n; ++i)
      ans += Math.min(l[i], r[i]) - height[i];

    return ans;
  }
}

//multiply strings 
class Solution {
  public String multiply(String num1, String num2) {
    final int m = num1.length();
    final int n = num2.length();

    StringBuilder sb = new StringBuilder();
    int[] pos = new int[m + n];

    for (int i = m - 1; i >= 0; --i)
      for (int j = n - 1; j >= 0; --j) {
        final int multiply = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
        final int sum = multiply + pos[i + j + 1];
        pos[i + j] += sum / 10;
        pos[i + j + 1] = sum % 10;
      }

    for (final int p : pos)
      if (p > 0 || sb.length() > 0)
        sb.append(p);

    return sb.length() == 0 ? "0" : sb.toString();
  }
}

//wildcard matching 
class Solution {
  public boolean isMatch(String s, String p) {
    final int m = s.length();
    final int n = p.length();
    // dp[i][j] := true if s[0..i) matches p[0..j)
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;

    for (int j = 0; j < p.length(); ++j)
      if (p.charAt(j) == '*')
        dp[0][j + 1] = dp[0][j];

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        if (p.charAt(j) == '*') {
          final boolean matchEmpty = dp[i + 1][j];
          final boolean matchSome = dp[i][j + 1];
          dp[i + 1][j + 1] = matchEmpty || matchSome;
        } else if (isMatch(s, i, p, j)) {
          dp[i + 1][j + 1] = dp[i][j];
        }

    return dp[m][n];
  }

  private boolean isMatch(final String s, int i, final String p, int j) {
    return j >= 0 && p.charAt(j) == '?' || s.charAt(i) == p.charAt(j);
  }
}

//jump game 2
class Solution {
  public int jump(int[] nums) {
    int ans = 0;
    int end = 0;
    int farthest = 0;

    // Start an implicit BFS.
    for (int i = 0; i < nums.length - 1; ++i) {
      farthest = Math.max(farthest, i + nums[i]);
      if (farthest >= nums.length - 1) {
        ++ans;
        break;
      }
      if (i == end) {   // Visited all the items on the current level.
        ++ans;          // Increment the level.
        end = farthest; // Make the queue size for the next level.
      }
    }

    return ans;
  }
}
//permutations
class Solution {
  public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();

    dfs(nums, new boolean[nums.length], new ArrayList<>(), ans);
    return ans;
  }

  private void dfs(int[] nums, boolean[] used, List<Integer> path, List<List<Integer>> ans) {
    if (path.size() == nums.length) {
      ans.add(new ArrayList<>(path));
      return;
    }

    for (int i = 0; i < nums.length; ++i) {
      if (used[i])
        continue;
      used[i] = true;
      path.add(nums[i]);
      dfs(nums, used, path, ans);
      path.remove(path.size() - 1);
      used[i] = false;
    }
  }
}

//permuatations 2
class Solution {
  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(nums);
    dfs(nums, new boolean[nums.length], new ArrayList<>(), ans);
    return ans;
  }

  private void dfs(int[] nums, boolean[] used, List<Integer> path, List<List<Integer>> ans) {
    if (path.size() == nums.length) {
      ans.add(new ArrayList<>(path));
      return;
    }

    for (int i = 0; i < nums.length; ++i) {
      if (used[i])
        continue;
      if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])
        continue;
      used[i] = true;
      path.add(nums[i]);
      dfs(nums, used, path, ans);
      path.remove(path.size() - 1);
      used[i] = false;
    }
  }
}

//rotate image
class Solution {
  public void rotate(int[][] matrix) {
    for (int i = 0, j = matrix.length - 1; i < j; ++i, --j) {
      int[] temp = matrix[i];
      matrix[i] = matrix[j];
      matrix[j] = temp;
    }

    for (int i = 0; i < matrix.length; ++i)
      for (int j = i + 1; j < matrix.length; ++j) {
        final int temp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = temp;
      }
  }
}

//group anagrams 
class Solution {
  public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> keyToAnagrams = new HashMap<>();

    for (final String str : strs) {
      char[] chars = str.toCharArray();
      Arrays.sort(chars);
      String key = String.valueOf(chars);
      keyToAnagrams.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
    }

    return new ArrayList<>(keyToAnagrams.values());
  }
}

//Powx,n
class Solution {
  public double myPow(double x, long n) {
    if (n == 0)
      return 1;
    if (n < 0)
      return 1 / myPow(x, -n);
    if (n % 2 == 1)
      return x * myPow(x, n - 1);
    return myPow(x * x, n / 2);
  }
}
