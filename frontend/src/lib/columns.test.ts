import { describe, expect, it } from "vitest";
import { diffColumns, parseCsvHeader } from "./columns";

describe("parseCsvHeader", () => {
  it("splits the first line into trimmed column names", () => {
    expect(parseCsvHeader("a, b ,c\n1,2,3")).toEqual(["a", "b", "c"]);
  });

  it("strips surrounding quotes", () => {
    expect(parseCsvHeader('"a","b"\n1,2')).toEqual(["a", "b"]);
  });

  it("returns an empty list for an empty file", () => {
    expect(parseCsvHeader("")).toEqual([]);
  });
});

describe("diffColumns", () => {
  it("returns null when the columns match, regardless of order", () => {
    expect(diffColumns(["a", "b"], ["b", "a"])).toBeNull();
  });

  it("names columns missing from the new file", () => {
    expect(diffColumns(["a", "b", "c"], ["a", "b"])).toEqual({
      missing: ["c"],
      unexpected: [],
    });
  });

  it("names columns the new file has that training never saw", () => {
    expect(diffColumns(["a", "b"], ["a", "b", "d"])).toEqual({
      missing: [],
      unexpected: ["d"],
    });
  });

  it("reports both missing and unexpected together", () => {
    expect(diffColumns(["a", "b"], ["a", "c"])).toEqual({
      missing: ["b"],
      unexpected: ["c"],
    });
  });
});
