import {
  Box,
  Button,
  Container,
  Flex,
  Grid,
  GridItem,
  Heading,
  Icon,
  Spacer,
  Text,
  VStack,
} from "@chakra-ui/react";
import type { NextPage } from "next";
import { useRouter } from "next/router";
import { useState } from "react";
import type { IconType } from "react-icons";
import { BiHeart, BiHeartSquare, BiRightArrowAlt } from "react-icons/bi";
import Layout from "../components/layout";

interface ApplicationPanelProps {
  icon: IconType;
  title: string;
  description: string;
  target: string;
  // full is 5
  colSpan: number;
}

const ApplicationPanel = ({
  icon,
  title,
  description,
  target,
  colSpan,
}: ApplicationPanelProps) => {
  const router = useRouter();
  const [hovered, setHovered] = useState(false);

  return (
    <GridItem colSpan={{ base: 1, md: colSpan }}>
      <Flex
        h="100%"
        direction="column"
        borderWidth={1}
        borderStyle="solid"
        borderColor="secondary.400"
        borderRadius="2xl"
        background={hovered ? "secondary.50" : undefined}
        boxShadow="lg"
        p={6}
        cursor="pointer"
        onClick={() => router.push(target)}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <VStack alignItems="flex-start" mb={6}>
          <Icon as={icon} color="secondary.500" fontSize={52} mb={2} />
          <Heading
            as="h2"
            fontSize={30}
            color="secondary.400"
            fontWeight="semibold"
            pb={6}
          >
            {title}
          </Heading>
          <Text color={hovered ? "secondary.400" : undefined}>
            {description}
          </Text>
        </VStack>

        <Spacer />

        <Flex flex={1} justifyContent="flex-end" alignItems="flex-end">
          <Button
            variant="outline"
            colorScheme="secondary"
            color="secondary.400"
            rightIcon={
              <Icon as={BiRightArrowAlt} color="primary.300" fontSize={20} />
            }
          >
            ใช้งาน
          </Button>
        </Flex>
      </Flex>
    </GridItem>
  );
};

const Home: NextPage = () => {
  return (
    <Layout>
      {/* top panel */}
      <Box>
        <Flex
          marginLeft="calc(50% - 50vw)"
          width="100vw"
          backgroundColor="secondary.50"
        >
          <Container
            maxW="container.lg"
            backgroundImage="/images/landing-ecg-signal.svg"
            backgroundSize="auto 80%"
            backgroundRepeat="no-repeat"
            backgroundPosition="right"
          >
            <VStack
              maxW={{ base: "100%", md: "55%", lg: "45%" }}
              alignItems="flex-start"
              py={20}
            >
              <Heading
                as="h4"
                fontWeight="semibold"
                size="md"
                color="primary.300"
              >
                หทัย AI
              </Heading>
              <Heading
                as="h1"
                fontWeight="semibold"
                fontSize={40}
                lineHeight="tall"
                color="secondary.400"
                pb={2}
              >
                AI ทำนายผล
                <br />
                ภาพสแกน
                <br />
                คลื่นไฟฟ้าหัวใจ
                <br />
              </Heading>
              <Text>
                AI ทำนายความน่าจะเป็นของการมี
                <Text as="span" color="primary.300">
                  รอยแผลเป็นในหัวใจ
                </Text>
                และ
                <Text as="span" color="primary.300">
                  ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้าย
                </Text>
                จากภาพสแกนคลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG) แบบ 12 Lead
              </Text>
            </VStack>
          </Container>
        </Flex>
      </Box>

      {/* application panels */}
      <Box pt={10} pb={20}>
        <Grid
          templateColumns={{ base: "repeat(1, 1fr)", md: "repeat(5, 1fr)" }}
          gap={4}
        >
          <ApplicationPanel
            icon={BiHeart}
            title="AI ทำนายผลภาพสแกน"
            description="AI ทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจ และค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายจากภาพสแกนคลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG) แบบ 12 Lead"
            colSpan={3}
            target="/ecg"
          />
          <ApplicationPanel
            icon={BiHeartSquare}
            title="แบบประเมินความเสี่ยงภาวะโรคหัวใจต่าง ๆ"
            description="แบบประเมินความเสี่ยงของการมีรอยแผลเป็นในหัวใจ (Myocardial Scar) โรคหลอดเลือดแดงของหัวใจตีบหรือตัน (Coronary Artery Disease, CAD) และการบีบตัวของหัวใจห้องล่างซ้ายผิดปกติ (LVEF) จากสถิติของประชากรผู้ป่วย สำหรับประชาชนทั่วไปหรือผู้ที่มีความเสี่ยง"
            colSpan={2}
            target="/assessment"
          />
        </Grid>
      </Box>
    </Layout>
  );
};

export default Home;
