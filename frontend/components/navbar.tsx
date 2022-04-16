import { FaBars, FaTimes, FaAngleDown, FaAngleRight } from "react-icons/fa";
import { BiUserCircle } from "react-icons/bi";
import {
  Avatar,
  Box,
  Button,
  Collapse,
  Flex,
  Icon,
  IconButton,
  Image,
  Link,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Stack,
  Text,
  useBreakpointValue,
  useColorModeValue,
  useDisclosure,
} from "@chakra-ui/react";
import NextLink from "next/link";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

export interface NavItem {
  label: string;
  subLabel?: string;
  children?: Array<NavItem>;
  href?: string;
}

interface WithSubnavigationProps {
  onClickLogin: () => void;
  onClickLogout: () => void;
  isLoggedIn: boolean;
  isLoadingUserStatus: boolean;
}

const BRAND: string = "หทัย AI";
const SIGN_IN_LABEL: string = "เข้าสู่ระบบ";
const NAV_ITEMS: Array<NavItem> = [
  {
    label: "อ่านคลื่นไฟฟ้าหัวใจ",
    href: "/ecg",
  },
  {
    label: "แบบประเมินความเสี่ยง",
    href: "/assessment",
  },
  {
    label: "เกี่ยวกับเรา",
    href: "/about",
  },
];

export default function WithSubnavigation({
  onClickLogin,
  onClickLogout,
  isLoggedIn,
  isLoadingUserStatus,
}: WithSubnavigationProps) {
  const router = useRouter();

  const { isOpen, onToggle } = useDisclosure();
  const signInButtonVariant = useBreakpointValue({
    base: "link",
    md: "outline",
  });

  return (
    <Box backgroundColor={"white"}>
      <Flex
        bg={useColorModeValue("white", "gray.800")}
        color={useColorModeValue("gray.600", "white")}
        minH={"60px"}
        p={{ base: 4 }}
        align={"center"}
        width={{ md: "container.md", lg: "container.lg" }}
        margin={"auto"}
      >
        <Flex
          flex={{ base: 1, md: "auto" }}
          ml={{ base: -2 }}
          display={{ base: "flex", md: "none" }}
        >
          <IconButton
            onClick={onToggle}
            icon={isOpen ? <FaTimes /> : <FaBars />}
            variant={"ghost"}
            aria-label={"Toggle Navigation"}
          />
        </Flex>
        <Flex flex={{ base: 1 }} justify={{ base: "center", md: "start" }}>
          <NextLink href="/" passHref>
            <Link _hover={{ textDecorationStyle: "none" }}>
              <Image
                src="/images/hatai-ai-logo.png"
                title="หทัย AI"
                alt="หทัย AI"
                height={10}
              />
              {/* <Text
                as="b"
                fontSize={{ base: "2xl", md: "3xl" }}
                textAlign={useBreakpointValue({ base: "center", md: "left" })}
                fontFamily={"heading"}
                color="pink.400"
              >
                {BRAND}
              </Text> */}
            </Link>
          </NextLink>

          <Flex display={{ base: "none", md: "flex" }} mx="auto">
            <DesktopNav />
          </Flex>
        </Flex>

        <Stack
          flex={{ base: 1, md: 0 }}
          justify={"flex-end"}
          direction={"row"}
          spacing={6}
        >
          {isLoadingUserStatus ? null : !isLoggedIn ? (
            <Button
              variant={signInButtonVariant}
              leftIcon={<BiUserCircle size="18px" />}
              fontSize={"sm"}
              fontWeight={600}
              colorScheme={"primary"}
              color="primary.300"
              px={3}
              onClick={() => {
                onClickLogin();
              }}
            >
              {SIGN_IN_LABEL}
            </Button>
          ) : (
            <Menu>
              <MenuButton>
                <Avatar size={"xs"} />
              </MenuButton>
              <MenuList
                position={"absolute"}
                right={-6}
                width="150px"
                minW="150px"
              >
                <MenuItem
                  justifyContent={"center"}
                  onClick={() => router.push("/user")}
                >
                  ข้อมูลการใช้งาน
                </MenuItem>
                <MenuItem justifyContent={"center"} onClick={onClickLogout}>
                  ออกจากระบบ
                </MenuItem>
              </MenuList>
            </Menu>
          )}
        </Stack>
      </Flex>

      <Collapse in={isOpen} animateOpacity>
        <MobileNav />
      </Collapse>
    </Box>
  );
}

const DesktopNav = () => {
  const linkColor = useColorModeValue("secondary.400", "gray.200");
  const linkHoverColor = useColorModeValue("secondary.600", "white");
  const popoverContentBgColor = useColorModeValue("white", "gray.800");

  const router = useRouter();
  const [currentPath, setCurrentPath] = useState("/");

  useEffect(() => {
    setCurrentPath(router.pathname);
  }, [router]);

  return (
    <Stack direction={"row"} spacing={4} alignItems="center">
      {NAV_ITEMS.map((navItem) => (
        <Box key={navItem.label}>
          {navItem.href ? (
            <NextLink href={navItem.href ?? "#"} passHref>
              <Link
                p={2}
                fontSize={"sm"}
                fontWeight={500}
                color={linkColor}
                _hover={{
                  textDecoration: "none",
                  color: linkHoverColor,
                }}
              >
                <Box
                  as="span"
                  fontWeight={
                    currentPath === navItem.href ? "semibold" : "regular"
                  }
                  borderBottomWidth={currentPath === navItem.href ? 2 : 0}
                  borderBottomStyle="solid"
                  borderColor="secondary.400"
                >
                  {navItem.label}
                </Box>
              </Link>
            </NextLink>
          ) : (
            <Popover trigger={"hover"} placement={"bottom-start"}>
              <PopoverTrigger>
                <Link
                  href={navItem.href ?? "#"}
                  p={2}
                  fontSize={"sm"}
                  fontWeight={500}
                  color={linkColor}
                  _hover={{
                    textDecoration: "none",
                    color: linkHoverColor,
                  }}
                >
                  {navItem.label}
                </Link>
              </PopoverTrigger>

              {navItem.children && (
                <PopoverContent
                  border={0}
                  boxShadow={"xl"}
                  bg={popoverContentBgColor}
                  p={4}
                  rounded={"xl"}
                  minW={"sm"}
                >
                  <Stack>
                    {navItem.children.map((child) => (
                      <DesktopSubNav key={child.label} {...child} />
                    ))}
                  </Stack>
                </PopoverContent>
              )}
            </Popover>
          )}
        </Box>
      ))}
    </Stack>
  );
};

const DesktopSubNav = ({ label, href, subLabel }: NavItem) => {
  return (
    <Link
      href={href || "#"}
      role={"group"}
      display={"block"}
      p={2}
      rounded={"md"}
      _hover={{ bg: useColorModeValue("pink.50", "gray.900") }}
    >
      <Stack direction={"row"} align={"center"}>
        <Box>
          <Text
            transition={"all .3s ease"}
            _groupHover={{ color: "pink.400" }}
            fontWeight={500}
          >
            {label}
          </Text>
          <Text fontSize={"sm"}>{subLabel}</Text>
        </Box>
        <Flex
          transition={"all .3s ease"}
          transform={"translateX(-10px)"}
          opacity={0}
          _groupHover={{ opacity: "100%", transform: "translateX(0)" }}
          justify={"flex-end"}
          align={"center"}
          flex={1}
        >
          <Icon color={"pink.400"} w={5} h={5} as={FaAngleRight} />
        </Flex>
      </Stack>
    </Link>
  );
};

const MobileNav = () => {
  return (
    <Stack
      bg={useColorModeValue("white", "gray.800")}
      p={4}
      display={{ md: "none" }}
    >
      {NAV_ITEMS.map((navItem) => (
        <MobileNavItem key={navItem.label} {...navItem} />
      ))}
    </Stack>
  );
};

const MobileNavItem = ({ label, children, href }: NavItem) => {
  const { isOpen, onToggle } = useDisclosure();

  return (
    <Stack spacing={4} onClick={children && onToggle}>
      <Flex
        py={2}
        as={Link}
        href={href ?? "#"}
        justify={"space-between"}
        align={"center"}
        _hover={{
          textDecoration: "none",
        }}
      >
        <Text
          fontWeight={600}
          color={useColorModeValue("secondary.400", "gray.200")}
        >
          {label}
        </Text>
        {children && (
          <Icon
            as={FaAngleDown}
            transition={"all .25s ease-in-out"}
            transform={isOpen ? "rotate(180deg)" : ""}
            w={6}
            h={6}
          />
        )}
      </Flex>

      <Collapse in={isOpen} animateOpacity style={{ marginTop: "0!important" }}>
        <Stack
          mt={2}
          pl={4}
          borderLeft={1}
          borderStyle={"solid"}
          borderColor={useColorModeValue("gray.200", "gray.700")}
          align={"start"}
        >
          {children &&
            children.map((child) => (
              <Link key={child.label} py={2} href={child.href || "#"}>
                {child.label}
              </Link>
            ))}
        </Stack>
      </Collapse>
    </Stack>
  );
};
