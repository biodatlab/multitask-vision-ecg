import {
  Heading,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  VStack,
} from "@chakra-ui/react";

const header = [
  "โมเดล",
  <>
    ความแม่นยำ
    <br />
    (Accuracy)
  </>,
  <>
    ความจำเพาะ
    <br />
    (Specificity)
  </>,
  <>
    ความไว
    <br />
    (Sensitivity)
  </>,
  "AUROC",
];
const rows = [
  ["แผลเป็นในหัวใจ (Mycardial scar)", 80.5, 85.0, 80.5, 89.1],
  ["LVEF < 40", 89.0, 88.6, 89.0, 92.9],
  ["LVEF < 50", 84.8, 84.2, 84.8, 90.5],
];

const ModelDescription = () => {
  return (
    <VStack gap={6} alignItems="flex-start" py={12} pb={16}>
      <Heading as="h5" fontWeight="semibold" size="md" color="secondary.400">
        วิธีการสร้างและวัดผลความแม่นยำของโมเดล
      </Heading>
      <VStack gap={2} fontSize="sm" px={6}>
        <Text>
          โมเดลทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG) แบบ 12
          Lead ที่ใช้ทำนายนี้ถูกสร้างขึ้นมาจากฐานข้อมูลภาพสแกนคลื่นไฟฟ้า
          หัวใจจำนวน 2,100 ภาพ ที่เก็บจากศูนย์โรคหัวใจโรงพยาบาลศิริราช
          โดยเป็นข้อมูลของคลื่นไฟฟ้าหัวใจที่ไม่มีแผลเป็น 1,100 ภาพ และมีแผลเป็น
          1,000 ภาพ
        </Text>

        <Text>
          ในการสร้างโมเดลการทำนาย ข้อมูลถูกแบ่งเป็นชุดข้อมูลสำหรับสร้างโมเดล,
          วัดผลระหว่างคำนวณโมเดล และทดสอบโมเดล ด้วยสัดส่วน 80:10:10
          ทั้งนี้โมเดลอาจมีความผิดพลาดในการทำนายผลได้
          ผู้ใช้งานสามารถดูความถูกต้องของการทดสอบโมเดลเบื้องต้นได้ตามตารางด้านล่าง
          (ผลเป็น %)
        </Text>
      </VStack>

      <Heading as="h5" fontWeight="semibold" size="md" color="secondary.400">
        การวัดผลความแม่นยำของโมเดล
      </Heading>
      <TableContainer
        backgroundColor="white"
        borderRadius="2xl"
        p={2}
        pt={4}
        sx={{
          marginX: "auto !important",
        }}
      >
        <Table variant="simple">
          <Thead>
            <Tr>
              {header.map((item, index) => {
                if (index === 0) {
                  return <Th key={String(Math.random())}>{item}</Th>;
                }

                return (
                  <Th key={String(Math.random())} isNumeric>
                    <Text textAlign="center" color="gray.600">
                      {item}
                    </Text>
                  </Th>
                );
              })}
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((row, rowsInd) => {
              return (
                <Tr key={String(Math.random())}>
                  {row.map((item, rowInd) => {
                    if (rowInd === 0) {
                      return (
                        <Td
                          key={String(Math.random())}
                          borderBottom={
                            rowsInd === rows.length - 1 ? "none" : undefined
                          }
                        >
                          {item}
                        </Td>
                      );
                    }

                    return (
                      <Td
                        key={String(Math.random())}
                        borderBottom={
                          rowsInd === rows.length - 1 ? "none" : undefined
                        }
                        isNumeric
                      >
                        <Text textAlign="center" color="gray.600">
                          {typeof item === "number" ? item.toFixed(1) : item}
                        </Text>
                      </Td>
                    );
                  })}
                </Tr>
              );
            })}
          </Tbody>
        </Table>
      </TableContainer>
    </VStack>
  );
};

export default ModelDescription;
