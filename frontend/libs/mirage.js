import { createServer } from "miragejs";

const TITLE_DESC_MAP = {
  scar: {
    title: "Myocardial Scar",
    description: "ความน่าจะเป็นที่จะมีแผลเป็นที่กล้ามเนื้อหัวใจ",
    average: 47.62,
  },
  lvef40: {
    title: "LVEF < 40",
    description:
      "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 40 %",
    average: 59.47,
  },
  lvef50: {
    title: "LVEF < 50",
    description:
      "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 50 %",
    average: 59.47,
  },
};

const calculateRiskLevel = (prob) => {
  if (prob < 30) {
    return "ต่ำ";
  }
  if (prob >= 30 && prob < 70) {
    return "ปานกลาง";
  }

  return "สูง";
};

export function makeServer({ environment = "test" } = {}) {
  return createServer({
    environment,

    routes() {
      this.namespace = "api";

      this.post("/predict", () => {
        return ["scar", "lvef40", "lvef50"].map((modelName) => {
          const probability = Math.floor(Math.random() * 100)
          const titleDesc = TITLE_DESC_MAP[modelName];

          return {
            "title": titleDesc.title,
            "description": titleDesc.description,
            "average": titleDesc.average,
            "probability": probability,
            "risk_level": calculateRiskLevel(probability)
          }
        })
      });

      /**
       * @see https://github.com/vercel/next.js/issues/16874
       */
      // method 1
      // this.namespace = ""
      // this.passthrough()

      // method 2
      this.passthrough((req) => {
        if (req.url?.includes("/_next/")) {
          return true;
        }
      });
    },
  });
}
