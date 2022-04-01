import {
  Heading,
  Text,
  Stack,
  OrderedList,
  ListItem,
  Box,
} from "@chakra-ui/react";
import Layout from "../components/layout";

const Manual = () => {
  return (
    <Layout>
      <Stack direction="column" gap={2} pt={6}>
        <Heading as="h1">วิธีการใช้งาน</Heading>
        <Heading as="h2" size="md">
          ทำนายผลคลื่นไฟฟ้าหัวใจ
        </Heading>
        <Box>
          <OrderedList pl={4}>
            <ListItem>
              อัพโหลดหรือลากไฟล์ที่ต้องการไปยังพื้นที่
              ผู้ใช้สามารถกดกากบาทที่มุมขวาบนเพื่อลบไฟล์เพื่อเลือกใหม่ได้
              <br />
              [ภาพ]
            </ListItem>
            <ListItem>กดปุ่มทำนาย และรอโมเดลทำนายผล</ListItem>
            <ListItem>
              อ่านผลการทำนายใต้ภาพ
              <br />
              [ภาพ]
            </ListItem>
          </OrderedList>
        </Box>
        <Heading as="h2" size="md">
          แบบประเมินความเสี่ยงภาวะโรคหัวใจล้มเหลวในระยะเวลา 3 ปี
        </Heading>
        <Box>
          <OrderedList pl={4}>
            <ListItem>
              กรอกข้อมูลผ่านแบบฟอร์ม ได้แก่ อายุ เพศ และประวัติโรคต่่างๆ
              <br />
              [ภาพ]
            </ListItem>
            <ListItem>กดปุ่มทำนายผลเพื่อวัดความเสี่ยง</ListItem>
            <ListItem>
              อ่านผลการทำนาย ความเสี่ยงแบ่งได้เป็น 4 ระดับ
              <br />
              (1) ความเสี่ยงต่ำ ผลการทำนายน้อยกว่า 3%
              <br />
              (2) ความเสี่ยงปานกลาง ผลการทำนายระหว่าง 3 ถึง 5 %
              <br />
              (3) ความเสียงสูง ผลการทำนายระหว่่าง 5 ถึง 10 %
              <br />
              (4) ความเสี่ยงสูงมาก ผลการทำนายมากกว่า 10%
              <br />
              [ภาพ]
            </ListItem>
          </OrderedList>
        </Box>
      </Stack>
    </Layout>
  );
};

export default Manual;
