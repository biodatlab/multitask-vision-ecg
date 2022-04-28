import type { NextApiRequest, NextApiResponse } from "next";
import formula from "../../data/formula";
import type { prediction } from "../ecg";

// -- FUNCTIONS
const getRiskLabel = (percent: number) => {
  if (percent < 30) {
    return "ต่ำ";
  }

  if (percent >= 30 && percent < 70) {
    return "ปานกลาง";
  }

  return "สูง";
};

const snakeToCamel = (str: string): string =>
  str.toLowerCase().replace(/(_\w)/g, (m) => m.toUpperCase().substr(1));

// -- API
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    res.status(405).send({ message: "Only POST requests allowed" });
    return;
  }

  const {
    selectedDiseases,
    data,
  }: {
    selectedDiseases: Array<string>;
    data: { [key: string]: number };
  } = req.body;

  console.log("from /api route", {
    selectedDiseases,
    data,
  });

  const dataCamelCase = Object.entries(data).reduce(
    (acc, [key, value]) => ({
      ...acc,
      [snakeToCamel(key)]: value,
    }),
    {}
  );

  const results = selectedDiseases.reduce<Array<prediction>>((acc, cur) => {
    let logit;
    let title = "";
    let description = "";

    if (cur === "scar") {
      logit = formula.scar(dataCamelCase);
      title = "Myocardial Scar";
      description = "ความน่าจะเป็นที่จะมีแผลเป็นที่กล้ามเนื้อหัวใจ";
    }
    if (cur === "cadScar") {
      logit = formula.cadScar(dataCamelCase);
      title = "Coronary Artery Disease (CAD)";
      description = "ความน่าจะเป็นของการโรคหลอดแดงของหัวใจตีบหรือตัน";
    }
    if (cur === "lvef40") {
      logit = formula.lvef40(dataCamelCase);
      title = "LVEF < 40";
      description =
        "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 40%";
    }
    if (cur === "lvef50") {
      logit = formula.lvef50(dataCamelCase);
      title = "LVEF < 50";
      description =
        "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 50%";
    }

    const prob = formula.logitToProb(logit);

    return [
      ...acc,
      {
        title,
        description,
        risk_level: getRiskLabel(prob),
        probability: prob,
      },
    ];
  }, [] as Array<prediction>);

  // return results
  res.status(200).json(results);
}
